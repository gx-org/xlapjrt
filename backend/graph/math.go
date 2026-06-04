// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package graph

import (
	"github.com/gomlx/gopjrt/xlabuilder"
	"github.com/gx-org/backend/ops"
)

// Math returns the builder to build operations from the math package.
func (g *Graph) Math() ops.MathBuilder {
	return g
}

// Abs returns the absolute value of x.
func (g *Graph) Abs(x ops.Node) (ops.Node, error) {
	return g.UnaryFunc(x, xlabuilder.Abs)
}

// Ceil returns the ceiling of x.
func (g *Graph) Ceil(x ops.Node) (ops.Node, error) {
	return g.UnaryFunc(x, xlabuilder.Ceil)
}

// Cos returns a node computing the cosine.
func (g *Graph) Cos(x ops.Node) (ops.Node, error) {
	return g.UnaryFunc(x, xlabuilder.Cos)
}

// Erf returns the error function of x.
func (g *Graph) Erf(x ops.Node) (ops.Node, error) {
	return g.UnaryFunc(x, xlabuilder.Erf)
}

// Exp returns the computation for the exponential.
func (g *Graph) Exp(x ops.Node) (ops.Node, error) {
	return g.UnaryFunc(x, xlabuilder.Exp)
}

// Expm1 returns Exp(x)-1.
func (g *Graph) Expm1(x ops.Node) (ops.Node, error) {
	return g.UnaryFunc(x, xlabuilder.Expm1)
}

// Floor returns the floor of x.
func (g *Graph) Floor(x ops.Node) (ops.Node, error) {
	return g.UnaryFunc(x, xlabuilder.Floor)
}

// Log returns the natural logarithm of x.
func (g *Graph) Log(x ops.Node) (ops.Node, error) {
	return g.UnaryFunc(x, xlabuilder.Log)
}

// Log1p returns log(1+x).
func (g *Graph) Log1p(x ops.Node) (ops.Node, error) {
	return g.UnaryFunc(x, xlabuilder.Log1p)
}

// Logistic returns 1/(1+exp(-x)).
func (g *Graph) Logistic(x ops.Node) (ops.Node, error) {
	return g.UnaryFunc(x, xlabuilder.Logistic)
}

// Min returns the minimum between x and y.
func (g *Graph) Min(x, y ops.Node) (ops.Node, error) {
	return g.BinaryFunc(x, y, xlabuilder.Min)
}

// Max returns the maximum between x and y.
func (g *Graph) Max(x, y ops.Node) (ops.Node, error) {
	return g.BinaryFunc(x, y, xlabuilder.Max)
}

// Pow returns x to the power of y.
func (g *Graph) Pow(x, y ops.Node) (ops.Node, error) {
	return g.BinaryFunc(x, y, xlabuilder.Pow)
}

// Round returns the nearest integer of x.
func (g *Graph) Round(x ops.Node) (ops.Node, error) {
	return g.UnaryFunc(x, xlabuilder.Round)
}

// Rsqrt returns 1/sqrt(x).
func (g *Graph) Rsqrt(x ops.Node) (ops.Node, error) {
	return g.UnaryFunc(x, xlabuilder.Rsqrt)
}

// Sign returns the sign of x.
func (g *Graph) Sign(x ops.Node) (ops.Node, error) {
	return g.UnaryFunc(x, xlabuilder.Sign)
}

// Sin returns a node computing the sine.
func (g *Graph) Sin(x ops.Node) (ops.Node, error) {
	return g.UnaryFunc(x, xlabuilder.Sin)
}

// Sqrt returns sqrt(x).
func (g *Graph) Sqrt(x ops.Node) (ops.Node, error) {
	return g.UnaryFunc(x, xlabuilder.Sqrt)
}

// Tanh returns a node computing the hyperbolic tangent.
func (g *Graph) Tanh(x ops.Node) (ops.Node, error) {
	return g.UnaryFunc(x, xlabuilder.Tanh)
}
