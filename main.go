package main

import (
	"bitbucket.org/dtolpin/gogp/gp"
	"bitbucket.org/dtolpin/infergo/ad"
	"bitbucket.org/dtolpin/infergo/infer"
	"bitbucket.org/dtolpin/infergo/model"
	. "bitbucket.org/dtolpin/wigp/kernel/ad"
	. "bitbucket.org/dtolpin/wigp/model"
	. "bitbucket.org/dtolpin/wigp/priors/ad"
	"encoding/csv"
	"flag"
	"fmt"
	"gonum.org/v1/gonum/optimize"
	"gonum.org/v1/gonum/stat"
	"io"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"
)

var (
	ALG       = "lbfgs" // or "momentum", "adam"
	QUIET     = false
	NITER     = 1000 // number of iterations
	THRESHOLD = 1e-6 // gradient threshold
	RATE      = 0.01 // learning rate (for Adam)
	DECAY     = 0.99 // rate decay
	GAMMA     = 0.1  // momentum factor

)

func init() {
	flag.Usage = func() {
		fmt.Fprintf(flag.CommandLine.Output(),
			`A model with warped time. Invocation:
  %s [OPTIONS] < INPUT > OUTPUT
or
  %s [OPTIONS] selfcheck
In 'selfcheck' mode, the data hard-coded into the program is used,
to demonstrate basic functionality.
`, os.Args[0], os.Args[0])
		flag.PrintDefaults()
	}
	flag.StringVar(&ALG, "a", ALG, "optimization algorithm "+
		"(momentum, adam or lbfgs)")
	flag.BoolVar(&QUIET, "q", QUIET, "less output")
	flag.IntVar(&NITER, "n", NITER, "number of iterations")
	flag.Float64Var(&THRESHOLD, "t", THRESHOLD, "gradient threshold")
	flag.Float64Var(&RATE, "r", RATE, "learning rate (Momentum, Adam)")
	flag.Float64Var(&DECAY, "d", DECAY, "rate decay (Momentum)")
	flag.Float64Var(&GAMMA, "g", GAMMA, "momentum factor (Momentum)")
	rand.Seed(time.Now().UTC().UnixNano())
}

func main() {
	var (
		input  io.Reader = os.Stdin
		output io.Writer = os.Stdout
	)

	flag.Parse()
	switch {
	case flag.NArg() == 0:
	case flag.NArg() == 1 && flag.Arg(0) == "selfcheck":
		input = strings.NewReader(selfCheckData)
	default:
		panic("usage")
	}

	// Data
	// ----
	// Load the data
	var err error
	fmt.Fprint(os.Stderr, "loading...")
	X, Y, err := load(input)
	if err != nil {
		panic(err)
	}
	fmt.Fprintln(os.Stderr, "done")

	// Normalize Y
	meany, stdy := stat.MeanStdDev(Y, nil)
	for i := range Y {
		Y[i] = (Y[i] - meany) / stdy
	}

	// Problem
	// -------
	// Define the problem
	priors := &ARPriors{}
	gp := &gp.GP{
		NDim:  2,
		Simil: AR,
		Noise: Noise,
	}
	m := &Model{
		GP:     gp,
		Priors: priors,
	}
	nTheta := gp.Simil.NTheta() + gp.Noise.NTheta() + m.Priors.NTheta()

	// Inference
	// ---------
	// Forecast one step out of sample, iteratively.
	// Output data augmented with predictions.
	fmt.Fprintln(os.Stderr, "Forecasting...")
	fmt.Fprintf(output, "%s,%s,%s,%s,%s,%s,%s",
		"X", "Y", "mu", "sigma", "lambda", "ll0", "ll")
	for i := 0; i != nTheta; i++ {
		fmt.Fprintf(output, ",theta[%d]", i)
	}
	fmt.Fprintln(output)
	for end := 1; end != len(X); end++ {
		m.X = X[:end]
		m.Y = Y[:end]

		// Construct the initial point in the optimization space
		x := make([]float64, nTheta+end-1)
		for i := range x {
			x[i] = 0.1 * rand.NormFloat64()
		}

		// Initial log likelihood
		ll0 := m.Observe(x)
		model.DropGradient(m)

		// Optimize the parameters
		switch ALG {
		case "momentum", "adam":
			var opt infer.Grad
			switch ALG {
			case "momentum":
				opt = &infer.Adam{Rate: RATE}
			case "adam":
				opt = &infer.Momentum{
					Rate:  RATE,
					Decay: DECAY,
					Gamma: GAMMA,
				}
			}
			iter := 0
		Iters:
			for ; iter != NITER; iter++ {
				ll, grad := opt.Step(m, x)
				if !QUIET && (iter+1)%10 == 0 {
					fmt.Fprintf(os.Stderr, "%05d %10.3f\r", iter+1, ll)
				}
				for i := range grad {
					if math.Abs(grad[i]) >= THRESHOLD {
						continue Iters
					}
				}
				break Iters
			}
		case "lbfgs":
			Func, Grad := infer.FuncGrad(m)
			p := optimize.Problem{Func: Func, Grad: Grad}

			result, err := optimize.Minimize(
				p, x, &optimize.Settings{
					MajorIterations:   NITER,
					GradientThreshold: THRESHOLD,
					Concurrent:        0,
				}, nil)
			if err != nil && result.Stats.MajorIterations < 10 {
				fmt.Fprintf(os.Stderr,
					"%d: stuck after %d iterations: %v\n",
					end, result.Stats.MajorIterations, err)
			}
			x = result.X
		}

		// Final log likelihood
		ll := m.Observe(x)
		model.DropGradient(m)
		ad.DropAllTapes()

		// Forecast
		Z := [][]float64{{0}}
		Z[0][0] = gp.X[end-1][0] + X[end][0] - X[end-1][0]
		mu, sigma, err := m.GP.Produce(Z)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Failed to forecast: %v\n", err)
		}

		// Output forecasts
		fmt.Fprintf(output, "%f,", Z[0][0])
		fmt.Fprintf(output, "%f,%f,%f,%f,%f,%f",
			Y[end], mu[0], sigma[0], math.Exp(x[nTheta+end-2]), ll0, ll)
		for i := 0; i != nTheta; i++ {
			fmt.Fprintf(output, ",%f", math.Exp(x[i]))
		}
		fmt.Fprintln(output)
	}
	fmt.Fprintln(os.Stderr, "done")

	return
}

// load parses the data from csv and returns inputs and outputs,
// suitable for feeding to the GP.
func load(rdr io.Reader) (
	x [][]float64,
	y []float64,
	err error,
) {
	csv := csv.NewReader(rdr)
RECORDS:
	for {
		record, err := csv.Read()
		switch err {
		case nil:
			// record contains the data
			xi := make([]float64, len(record)-1)
			i := 0
			for ; i != len(record)-1; i++ {
				xi[i], err = strconv.ParseFloat(record[i], 64)
				if err != nil {
					// data error
					return x, y, err
				}
			}
			yi, err := strconv.ParseFloat(record[i], 64)
			if err != nil {
				// data error
				return x, y, err
			}
			x = append(x, xi)
			y = append(y, yi)
		case io.EOF:
			// end of file
			break RECORDS
		default:
			// i/o error
			return x, y, err
		}
	}

	return x, y, err
}

var selfCheckData = `0.1,-3.376024003717768007e+00
0.3,-1.977828720240523142e+00
0.5,-1.170229755402199645e+00
0.7,-9.583612412106726763e-01
0.9,-8.570477029219900622e-01
1.1,-8.907618364403485645e-01
1.3,-2.611461145416017482e-01
1.5,1.495844460881872728e-01
1.7,-4.165391766465373347e-01
1.9,-2.875013255153459069e-01
2.1,3.869524825854843142e-01
2.3,9.258652056784907325e-01
2.5,5.858145290237386504e-01
2.7,8.788023289396607041e-01
2.9,1.233057437482850682e+00
3.1,1.066540422694190138e+00
3.3,9.137144265931921305e-01
3.5,7.412075911286820640e-01
3.7,1.332146185234786673e+00
3.9,1.439962957400109378e+00
4.1,1.222960311200699257e+00
4.3,2.026371435028667956e-01
4.5,-1.659683673486037625e+00
4.7,-9.881392068563286113e-01
4.9,-3.948046844798779875e-01
5.1,-2.635420428119399916e-01
5.3,-1.610738281677652317e+00
5.5,-3.092358176820052540e-01
5.7,-2.958870744615414994e-01
5.9,-1.619124030623840138e+00
6.1,-1.241765328045226102e+00
6.3,-2.933200084576037536e-01
6.5,-6.066731986714126723e-01
6.7,5.866702176917204525e-01
6.9,6.282566869554838673e-01
7.1,1.013316587545910918e+00
7.3,1.123871563448763267e+00
7.5,1.094949286471081251e+00
7.7,1.113603299433020055e+00
7.9,8.567255613058102348e-01
8.1,7.384693873911447604e-01
8.3,3.434834982521656199e-01
8.5,-2.514717991306942083e-02
`
