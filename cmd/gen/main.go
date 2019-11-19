package main

import (
	"bitbucket.org/dtolpin/gogp/gp"
	"bitbucket.org/dtolpin/gogp/kernel"
	adkernel "bitbucket.org/dtolpin/gogp/kernel/ad"
	"bitbucket.org/dtolpin/infergo/ad"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"time"
)

var (
	SEASONAL = false
)

func init() {
	flag.Usage = func() {
		fmt.Fprintf(flag.CommandLine.Output(),
			`Generate test data. Invocation:
	%s  [OPTIONS] | head -100
`, os.Args[0])
		flag.PrintDefaults()
	}
	flag.BoolVar(&SEASONAL, "seasonal", SEASONAL, "seasonal kernel")
	rand.Seed(time.Now().UTC().UnixNano())
	ad.MTSafeOn()
}

type xkernel struct{}

var xKernel xkernel

const (
	xVariance    = 0.0625
	xLengthScale = 5.
)

func (xkernel) Observe(x []float64) float64 {
	const (
		xa = iota
		xb
	)

	return xVariance * kernel.Normal.Cov(xLengthScale, x[xa], x[xb])
}

func (xkernel) Gradient() []float64 {
	return []float64{1, 1}
}

func (xkernel) NTheta() int {
	return 0
}

const (
	yVariance            = 1.
	yLengthScale         = 10.
	yPeriod              = 10.
	ySeasonalVariance    = 1.
	ySeasonalLengthScale = 2.
)

type arkernel struct{}

var arKernel arkernel

func (arkernel) Observe(x []float64) float64 {
	const (
		xa = iota
		xb
	)

	return yVariance * kernel.Matern52.Cov(ySeasonalLengthScale, x[xa], x[xb])
}

func (arkernel) Gradient() []float64 {
	return []float64{1, 1}
}

func (arkernel) NTheta() int {
	return 0
}

type sarkernel struct{}

var sarKernel sarkernel

func (sarkernel) Observe(x []float64) float64 {
	const (
		xa = iota
		xb
	)

	return yVariance*kernel.Matern52.Cov(yLengthScale, x[xa], x[xb]) +
		ySeasonalVariance*kernel.Periodic.Cov(ySeasonalLengthScale, yPeriod, x[xa], x[xb])
}

func (sarkernel) Gradient() []float64 {
	return []float64{1, 1}
}

func (sarkernel) NTheta() int {
	return 0
}

func sample(g *gp.GP, xs <-chan float64, xys chan<- [2]float64) {
	for {
		x := <-xs
		X := [][]float64{{x}}
		Y, Sigma, err := g.Produce(X)
		if err != nil {
			panic(fmt.Errorf("produce: %v", err))
		}
		y := Y[0] + Sigma[0]*rand.NormFloat64()
		xys <- [...]float64{x, y}
		X = append(g.X, X...)
		Y = append(g.Y, y)
		if err := g.Absorb(X, Y); err != nil {
			panic(fmt.Errorf("absorb: %v", err))
		}
	}
}

func main() {
	flag.Parse()

	gLambda := &gp.GP{
		NDim:  1,
		Simil: xKernel,
	}

	grid := make(chan float64, 1)
	lambdas := make(chan [2]float64, 1)
	xs := make(chan float64, 1)
	xys := make(chan [2]float64, 1)

	// Sampling inputs
	go func() {
		for x := 0.; ; x++ {
			grid <- x
		}
	}()
	go sample(gLambda, grid, lambdas)
	go func() {
		x := 0.
		for lambda := range lambdas {
			xs <- x
			dx := math.Exp(lambda[1])
			x += dx
		}
	}()

	// Sampling outputs
	gy := &gp.GP{
		NDim:  1,
		Noise: adkernel.ConstantNoise(0.01),
	}
	if SEASONAL {
		gy.Simil = sarKernel
	} else {
		gy.Simil = arKernel
	}

	go sample(gy, xs, xys)
	for xy := range xys {
		fmt.Printf("%f %f\n", xy[0], xy[1])
	}
}
