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
	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/gopjrt/xlabuilder"
	pjrtgx "github.com/gx-org/xlapjrt"
)

// Iota creates a constant of the given shape with increasing numbers (starting from 0) on the given axis.
func (g *Graph) Iota(shape shapes.Shape, iotaAxis int) (compute.Value, error) {
	xlaOp, err := xlabuilder.Iota(g.builder, pjrtgx.ToShape(shape), iotaAxis)
	if err != nil {
		return nil, err
	}
	return g.newNode(xlaOp), nil
}

// ArgMinMax returns a new argmin/argmax node.
func (g *Graph) ArgMinMax(x compute.Value, axis int, outputDType dtypes.DType, isMin bool) (compute.Value, error) {
	xlaOp, err := xlabuilder.ArgMinMax(g.xlaHandle(x), axis, pjrtgx.ToPJDType(outputDType), isMin)
	if err != nil {
		return nil, err
	}
	return g.newNode(xlaOp), nil
}

// ReduceMax is a shortcut for Reduce with the proper computation
// and initial value to reduce x on the given axes, by taking the max value.
//
// If no axes are given, it reduces the full array.
func (g *Graph) ReduceMax(x compute.Value, axes ...int) (compute.Value, error) {
	xlaOp, err := xlabuilder.ReduceMax(g.xlaHandle(x), axes...)
	if err != nil {
		return nil, err
	}
	return g.newNode(xlaOp), nil
}

// ReduceSum sums over axes.
func (g *Graph) ReduceSum(x compute.Value, axes ...int) (compute.Value, error) {
	xlaOp, err := xlabuilder.ReduceSum(g.xlaHandle(x), axes...)
	if err != nil {
		return nil, err
	}
	return g.newNode(xlaOp), nil
}

// Transpose transposes the axes of x.
func (g *Graph) Transpose(x compute.Value, permutation ...int) (compute.Value, error) {
	xlaOp, err := xlabuilder.Transpose(g.xlaHandle(x), permutation...)
	if err != nil {
		return nil, err
	}
	return g.newNode(xlaOp), nil
}
