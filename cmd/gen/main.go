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
	yLengthScale         = 20.
	yPeriod              = 10.
	ySeasonalVariance    = 4.
	ySeasonalLengthScale = 5. 
)

type tkernel struct{}

var tKernel tkernel

func (tkernel) Observe(x []float64) float64 {
	const (
		xa = iota
		xb
	)

	return yVariance * kernel.Matern52.Cov(ySeasonalLengthScale, x[xa], x[xb])
}

func (tkernel) Gradient() []float64 {
	return []float64{1, 1}
}

func (tkernel) NTheta() int {
	return 0
}

type skernel struct{}

var sKernel skernel

func (skernel) Observe(x []float64) float64 {
	const (
		xa = iota
		xb
	)

	return ySeasonalVariance*kernel.Periodic.Cov(ySeasonalLengthScale, yPeriod, x[xa], x[xb])
}

func (skernel) Gradient() []float64 {
	return []float64{1, 1}
}

func (skernel) NTheta() int {
	return 0
}

func sample(g *gp.GP, xs <-chan float64, ys chan<- float64) {
	for {
		x := <-xs
		X := [][]float64{{x}}
		Y, Sigma, err := g.Produce(X)
		if err != nil {
			panic(fmt.Errorf("produce: %v", err))
		}
		y := Y[0] + Sigma[0]*rand.NormFloat64()
		ys <- y
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
	lambdas := make(chan float64, 1)
	xs := make(chan float64, 1)
	ys := make(chan float64, 1)
	// for the seasonal component
	xSs := make(chan float64, 1)
	ySs := make(chan float64, 1)

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
			x += math.Exp(lambda)
		}
	}()

	// Sampling outputs
	gy := &gp.GP{
		NDim:  1,
		Simil: tKernel,
		Noise: adkernel.ConstantNoise(0.01),
	}

	go sample(gy, xs, ys)

	// Seasonal component, if present, is added
	// on the 'unwarped' inputs so that the period
	// stays fixed. 
	if SEASONAL {
		gyS := &gp.GP{
			NDim:  1,
			Simil: sKernel,
		}
		go sample(gyS, xSs, ySs)
	}

	x := 0.
	for y := range ys {
		if SEASONAL {
			xSs <- x
			yS := <- ySs
			y += yS
		}
		fmt.Printf("%f,%f\n", x, y)
		x++
	}
}
