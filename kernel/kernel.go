package kernel

import (
	"bitbucket.org/dtolpin/gogp/kernel"
)

// The autoregressive similarity kernel.
type ar struct{}

var AR ar

func (ar) Observe(x []float64) float64 {
	const (
		c   = iota // variance
		l          // length scale
		wxa        // first point
		_
		wxb // second point
		_
	)

	return x[c]*x[c] * kernel.Matern52.Cov(x[l], x[wxa], x[wxb])
}

func (ar) NTheta() int { return 2 }

// The seasonal+autoregressive similarity kernel. We pretend
// we know the period.
type SAR struct {
	Period float64
}

func (k *SAR) Observe(x []float64) float64 {
	const (
		c1  = iota // trend variance
		c2         // season variance
		l1         // trend length scale
		l2         // season length scale
		wxa        // first point
		xa
		wxb // second point
		xb
	)

	return x[c1]*x[c1]*kernel.Matern52.Cov(x[l1], x[wxa], x[wxb]) +
		// periodic kernel sees unwarped inputs
		x[c2]*x[c2]*kernel.Periodic.Cov(x[l2], k.Period, x[xa], x[xb])
}

func (SAR) NTheta() int { return 4 }

// The noise kernel
type noise struct{}

var Noise noise

func (n noise) Observe(x []float64) float64 {
	// The noise is scaled by 0.01 so that the initial value
	// log(s)=0 corresponds to standard deviation of 0.1.
	return 0.01 * kernel.UniformNoise.Observe(x)
}

func (noise) NTheta() int { return 1 }
