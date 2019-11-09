package kernel

import (
	"bitbucket.org/dtolpin/gogp/kernel"
)

// The similarity kernel.
type ysimil struct{}

var YSimil ysimil

func (ysimil) Observe(x []float64) float64 {
	const (
		c  = iota // output scale
		l         // length scale
		xa        // first point
		xb        // second point
	)

	return x[c] * kernel.Matern52.Cov(x[l], x[xa], x[xb])
}

func (ysimil) NTheta() int { return 2 }

// The X similarity kernel.
type xsimil struct{}

var XSimil xsimil

func (xsimil) Observe(x []float64) float64 {
	const (
		c  = iota // output scale
		l         // length scale
		xa        // first point
		xb        // second point
	)

	return x[c] * kernel.Normal.Cov(x[l], x[xa], x[xb])
}

func (xsimil) NTheta() int { return 2 }

// The noise kernel.
type ynoise struct{}

var YNoise ynoise

func (n ynoise) Observe(x []float64) float64 {
	return 0.1 * kernel.UniformNoise.Observe(x)
}

func (ynoise) NTheta() int { return 1 }

// The X noise kernel.
type xnoise struct{}

var XNoise xnoise

func (n xnoise) Observe(x []float64) float64 {
	return 0.1 * kernel.UniformNoise.Observe(x)
}

func (xnoise) NTheta() int { return 1 }
