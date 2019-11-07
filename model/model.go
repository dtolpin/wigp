package model

import (
	. "bitbucket.org/dtolpin/infergo/dist"
	"bitbucket.org/dtolpin/gogp/gp"
)

type Priors struct {
	X0     []float64 // original inputs
	GP     *gp.GP
}

func(m *Priors) NTheta() int {
	return m.GP.Simil.NTheta() + m.GP.Noise.NTheta()
}

func (m *Priors) Observe(x []float64) float64 {
	const (
		xc = iota // x output scale
		xl        // x length scale
		c         // output scale
		l         // length scale
		s         // noise
		i0        // first input
	)

	n := len(x[i0:]) / 2
	if len(m.X0) != n {
		if n > 0 {
			// First call, memoize initial distances between inputs
			m.X0 = make([]float64, n)
			copy(m.X0, x[i0:])
		} else {
			m.X0 = nil
		}
	}

	ll := 0.

	// Output scale is mostly less than 1.
	ll += Normal.Logp(-1, 1, x[c])
	ll += Normal.Logp(-1, 1, x[xc])

	// Length scale is around 1, in wide margins.
	ll += Normal.Logp(0, 2, x[l])
	ll += Normal.Logp(0, 2, x[xl])

	// The noise is scaled by 0.01 in the kernel.
	ll += Normal.Logp(0, 1, x[s])

	//  We allow inputs to move slightly.
	// TODO

	return ll
}
