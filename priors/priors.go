package priors

import (
	. "bitbucket.org/dtolpin/infergo/dist"
	"math"
)

type Priors struct {
}

func (m *Priors) NTheta() int {
	return 1
}

func (m *Priors) Observe(x []float64) float64 {
	const (
		c  = iota // variance
		l         // length scale
		s         // noise variance
		t		  // warping
		i0        // first transformation (relative step change)
	)

	ll := 0.

	// Priors of the Gaussian process
	// ------------------------------
	// Variance is mostly less than 1.
	ll += Normal.Logp(-1, 1, x[c])
	// Length scale is around 1, in wide margins.
	ll += Normal.Logp(-2, 2, x[l])
	// Noise variance is around 0.01, scaled in the kernel.
	ll += Normal.Logp(-0.5, 1, x[s])

	// Priors of the renewal process
	// -----------------------------
	//  We allow the inputs to move somewhat.
	ll += Normal.Logp(-0.5, 1/(1 + float64(len(x[i0:]))), x[t])
	ll += Normal.Logps(-0.125, math.Exp(x[t]), x[i0:]...)

	return ll
}
