package main

import (
	"bitbucket.org/dtolpin/gogp/gp"
	. "bitbucket.org/dtolpin/wigp/kernel/ad"
	. "bitbucket.org/dtolpin/wigp/model/ad"
	"bitbucket.org/dtolpin/infergo/ad"
	"bitbucket.org/dtolpin/infergo/infer"
	"bitbucket.org/dtolpin/infergo/model"
	"encoding/csv"
	"flag"
	"fmt"
	"gonum.org/v1/gonum/optimize"
	"io"
	"math"
	"os"
	"strings"
	"strconv"
)

var (
	NITER          = 0
	EPS    float64 = 0
	NTASKS         = 0
	LOGSIGMA float64
	SHOWWARP = false
)

func init() {
	LOGSIGMA = math.Log(0.5)
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
	flag.IntVar(&NITER, "niter", NITER,
		"number of optimizer iterations")
	flag.IntVar(&NTASKS, "ntasks", NTASKS,
		"number of optimizer tasks")
	flag.Float64Var(&EPS, "eps", EPS,
		"gradient threshold for early stopping")
	flag.Float64Var(&LOGSIGMA, "logsigma", LOGSIGMA,
		"log standard deviation of relative step")
	flag.BoolVar(&SHOWWARP, "show-warp", SHOWWARP,
		"show warped inputs")
}

type Model struct {
	gp           *gp.GP
	priors       *Priors
	gGrad, pGrad []float64
}

func (m *Model) Observe(x []float64) float64 {
	var gll, pll float64
	xgp := x[m.priors.NTheta():]
	gll, m.gGrad = m.gp.Observe(xgp), model.Gradient(m.gp)
	pll, m.pGrad = m.priors.Observe(x), model.Gradient(m.priors)
	return gll + pll
}

func (m *Model) Gradient() []float64 {
	for i := range m.gGrad {
		m.pGrad[i + m.priors.NTheta()] += m.gGrad[i]
	}

	// Wipe gradients of the last input and all outputs
	iy0 := m.priors.NTheta() +
		m.gp.Simil.NTheta() + m.gp.Noise.NTheta() +
		len(m.gp.X)
	for i := iy0; i != len(m.pGrad); i++ {
		m.pGrad[i] = 0
	}

	return m.pGrad
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

	g := &gp.GP{
		NDim:  1,
		Simil: Simil,
		Noise: Noise,
	}
	m := &Model{
		gp:     g,
		priors: &Priors{
			GP: &gp.GP {
				NDim: 1,
				Simil: XSimil,
				Noise: XNoise,
			},
		},
	}
	theta := make([]float64, m.priors.NTheta() + g.Simil.NTheta()+g.Noise.NTheta())

	// Collect results in a buffer to patch with updated inputs

	if SHOWWARP {
		buffer := strings.Builder{}
		Evaluate(g, m, theta, input, &buffer)
		// Predict at updated inputs
		mu, sigma, _ := g.Produce(g.X)

		// Patch
		lines := strings.Split(buffer.String(), "\n")
		ilast := len(lines) - 1
		if len(lines[ilast]) == 0 {
			// There is an extra empty line
			ilast--
		}
		for i, line := range lines[:ilast] {
			if len(line) == 0 {
				break
			}
			fields := strings.SplitN(line, ",", 5)
			fmt.Fprintf(output, "%f,%f,%f,%f,%s\n",
				g.X[i][0], g.Y[i], mu[i], sigma[i],
				fields[len(fields)-1])
		}

		// The last input is fixed, and the last line is left
		// unmodified
		fmt.Fprintln(output, lines[ilast])
	} else {
		Evaluate(g, m, theta, input, output)
	}
}

// Evaluate evaluates Gaussian process on CSV data.  One step
// out of sample forecast is recorded for each time point, along
// with the hyperparameters.  For optimization, LBFGS from the
// gonum library (http://gonum.org) is used for faster
// execution. In general though, LBFGS is a bit of hit-or-miss,
// failing to optimize occasionally, so in real applications a
// different optimization/inference algorithm may be a better
// choice.
func Evaluate(
	g *gp.GP, // gaussian process
	m model.Model, // optimization model
	theta []float64, // initial values of hyperparameters
	rdr io.Reader, // data
	wtr io.Writer, // forecasts
) error {
	// Load the data
	var err error
	fmt.Fprint(os.Stderr, "loading...")
	X, Y, err := load(rdr)
	if err != nil {
		return err
	}
	fmt.Fprintln(os.Stderr, "done")

	// Forecast one step out of sample, iteratively.
	// Output data augmented with predictions.
	fmt.Fprintln(os.Stderr, "Forecasting...")
	for end := 0; end != len(X); end++ {
		Xi := X[:end]
		Yi := Y[:end]

		// Construct the initial point in the optimization space
		var x []float64
		// The inputs are optimized as well as the
		// hyperparameters. The inputs are appended to the
		// parameter vector of Observe.
		x = make([]float64, len(theta)+len(Xi)*(g.NDim+1))
		copy(x, theta)
		k := len(theta)
		for j := range Xi {
			copy(x[k:], Xi[j])
			k += g.NDim
		}
		copy(x[k:], Yi)

		// Optimize the parameters
		Func, Grad := infer.FuncGrad(m)
		p := optimize.Problem{Func: Func, Grad: Grad}

		// Initial log likelihood
		lml0 := m.Observe(x)
		model.DropGradient(m)

		result, err := optimize.Minimize(
			p, x, &optimize.Settings{
				MajorIterations:   NITER,
				GradientThreshold: EPS,
				Concurrent:        NTASKS,
			}, nil)
		// We do not need the optimizer to `officially'
		// converge, a few iterations usually bring most
		// of the improvement. However, in pathological
		// cases even a single iteration does not succeed,
		// and we want to report that.
		if err != nil && result.Stats.MajorIterations == 1 {
			// There was a problem and the optimizer stopped
			// on first iteration.
			fmt.Fprintf(os.Stderr, "Failed to optimize: %v\n", err)
		}
		x = result.X

		// Final log likelihood
		lml := m.Observe(x)
		model.DropGradient(m)
		ad.DropAllTapes()

		// Forecast
		Z := X[end : end+1]
		mu, sigma, err := g.Produce(Z)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Failed to forecast: %v\n", err)
		}

		// Output forecasts
		z := Z[0]
		for j := range z {
			fmt.Fprintf(wtr, "%f,", z[j])
		}
		fmt.Fprintf(wtr, "%f,%f,%f,%f,%f",
			Y[end], mu[0], sigma[0], lml0, lml)
		for i := 0; i != len(theta); i++ {
			fmt.Fprintf(wtr, ",%f", math.Exp(x[i]))
		}
		fmt.Fprintln(wtr)
	}
	fmt.Fprintln(os.Stderr, "done")

	return nil
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
