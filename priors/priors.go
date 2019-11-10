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
		c  = iota // output scale
		l         // length scale
		s         // noise
		t         // standard deviation of the relative step
		i0        // first transformation (relative step change)
	)

	ll := 0.

	// Priors of the Gaussian process
	// ------------------------------
	// Output scale is mostly less than 1.
	ll += Normal.Logp(-1, 1, x[c])
	// Length scale is around 1, in wide margins.
	ll += Normal.Logp(0, 2, x[l])
	// The noise is scaled by 0.01 in the kernel.
	ll += Normal.Logp(0, 1, x[s])

	// Priors of the renewal process
	// -----------------------------
	//  We allow inputs to move slightly.
	ll += Normal.Logp(0, 10, x[t])
	ll += Normal.Logps(0, math.Exp(x[t]), x[i0:]...)

	return ll
}
