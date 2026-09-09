// Copyright 2026 Google LLC
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
	"github.com/pkg/errors"
	"github.com/gomlx/gopjrt/xlabuilder"
	"github.com/gx-org/backend"
)

// Shape returns the builder to build operations from the shape package.
func (g *Graph) Shape() backend.ShapeBuilder {
	return g
}

// Split implements the split operation in terms of slice, with static indices.
func (g *Graph) Split(x backend.Node, axis int, numSplits int) (backend.Node, error) {
	shap := x.(pjrtNode).BackendShape()
	rank := len(shap.AxisLengths)

	if axis < 0 || axis >= rank {
		return nil, errors.Errorf("axis %d is out of bounds for rank %d", axis, rank)
	}
	if shap.AxisLengths[axis]%numSplits != 0 {
		return nil, errors.Errorf("axis %d has size %d which is not divisible by %d numSplits", axis, shap.AxisLengths[axis], numSplits)
	}
	stride := shap.AxisLengths[axis] / numSplits
	slicedNodes := make([]backend.Node, numSplits)
	for i := range numSplits {
		starts := make([]int, rank)
		limits := make([]int, rank)
		strides := make([]int, rank)
		for axis, axisSize := range shap.AxisLengths {
			limits[axis] = axisSize
			strides[axis] = 1
		}

		starts[axis] = i * stride
		limits[axis] = i*stride + stride
		xlaOp, err := xlabuilder.Slice(g.xlaHandle(x), starts, limits, strides)
		if err != nil {
			return nil, err
		}

		slicedNodes[i] = g.newNode(xlaOp)
	}

	outputDims := append([]int{1}, shap.AxisLengths...)
	outputDims[axis+1] = stride

	reshapedNodes := make([]backend.Node, numSplits)
	for i := range slicedNodes {
		reshapedNode, err := g.Reshape(slicedNodes[i], outputDims)
		if err != nil {
			return nil, err
		}
		reshapedNodes[i] = reshapedNode
	}

	return g.Concat(0, reshapedNodes)
}

// Concat concatenates multiple arrays into a single array.
func (g *Graph) Concat(axis int, nodes []backend.Node) (backend.Node, error) {
	inputs, err := g.xlaHandles(nodes)
	if err != nil {
		return nil, err
	}

	xlaOp, err := xlabuilder.Concatenate(axis, inputs...)
	if err != nil {
		return nil, err
	}
	return g.newNode(xlaOp), nil
}
