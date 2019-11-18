package model

import (
	"bitbucket.org/dtolpin/gogp/gp"
	"bitbucket.org/dtolpin/infergo/model"
	. "bitbucket.org/dtolpin/wigp/kernel/ad"
	. "bitbucket.org/dtolpin/wigp/priors/ad"
	"math"
	"testing"
)

const (
	dx  = 1e-8
	eps = 1e-4
)

func TestGradient(t *testing.T) {
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

	for i, c := range []struct {
		x []float64
		X [][]float64
		Y []float64
	}{
		{
			x: []float64{0, 0, 0, 0, 0},
			X: [][]float64{{0}, {1}},
			Y: []float64{-0.3, 0.2},
		},
		{
			x: []float64{1, 1, 1, 1, 1},
			X: [][]float64{{0}, {1}},
			Y: []float64{-0.3, 0.3},
		},
		{
			x: []float64{0, 0, 0, 0, 0, 0},
			X: [][]float64{{0}, {1}, {2}},
			Y: []float64{-0.3, 0.2, -0.1},
		},
		{
			x: []float64{0, 0, 0, 0, 0, 0, 0},
			X: [][]float64{{0}, {1}, {2}, {3}},
			Y: []float64{-0.3, 0.2, -0.1, 0},
		},
	} {
		m.X = c.X
		m.Y = c.Y
		ll0 := m.Observe(c.x)
		grad := model.Gradient(m)
		for j := range c.x {
			x0 := c.x[j]
			c.x[j] += dx
			ll := m.Observe(c.x)
			dldx := (ll - ll0) / dx
			c.x[j] = x0
			if math.Abs(grad[j]-dldx) > eps {
				t.Errorf("%d: dl/dx%d mismatch: got %.8f, want %.4f",
					i, j, dldx, grad[j])
			}
		}
	}
}
