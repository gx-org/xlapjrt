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

// Cos returns a node computing the cosine.
func (g *Graph) Cos(x ops.Node) (ops.Node, error) {
	return g.UnaryFunc(x, xlabuilder.Cos)
}

// Sin returns a node computing the sine.
func (g *Graph) Sin(x ops.Node) (ops.Node, error) {
	return g.UnaryFunc(x, xlabuilder.Sin)
}

// Tanh returns a node computing the hyperbolic tangent.
func (g *Graph) Tanh(x ops.Node) (ops.Node, error) {
	return g.UnaryFunc(x, xlabuilder.Tanh)
}
