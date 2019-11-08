package model

import (
	. "bitbucket.org/dtolpin/infergo/dist"
)

type Priors struct{}

func (m *Priors) Observe(x []float64) float64 {
	const (
		cy = iota // y output scale
		ly        // y length scale
		sy        // y noise
		cx        // x output scale
		lx        // x length scale
		sx        // x noise
	)

	ll := 0.

	// Output scale is mostly less than 1.
	ll += Normal.Logp(-1, 1, x[cy])
	ll += Normal.Logp(-1, 1, x[cx])

	// Length scale is around 1, in wide margins.
	ll += Normal.Logp(0, 2, x[ly])
	ll += Normal.Logp(0, 2, x[lx])

	// The noise is scaled by 0.01 in the kernel.
	ll += Normal.Logp(0, 1, x[sy])
	ll += Normal.Logp(0, 1, x[sx])

	return ll
}
