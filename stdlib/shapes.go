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

package stdlib

import (
	"fmt"

	"github.com/gx-org/backend/graph"
	"github.com/gx-org/backend/shape"
	"github.com/gx-org/gx/api/values"
	"github.com/gx-org/gx/build/ir"
	"github.com/gx-org/gx/interp/elements"
	"github.com/gx-org/gx/interp"
	"github.com/gx-org/gx/interp/state"
)

func evalConcat(ctx interp.Context, call elements.CallAt, fn *elements.Func, irFunc *ir.FuncBuiltin, args []state.Element) (state.Element, error) {
	xs := make([]graph.Node, len(args)-1)
	xShapes := make([]*shape.Shape, len(args)-1)
	for i, arg := range args[1:] {
		var err error
		xs[i], xShapes[i], err = state.NodeFromElement(arg)
		if err != nil {
			return nil, err
		}
	}
	axis, err := elements.ConstantScalarFromElement[ir.Int](args[0])
	if err != nil {
		return nil, err
	}
	op, err := pjrtGraph(ctx).NewConcat(int(axis), xs)
	if err != nil {
		return nil, err
	}
	return ctx.State().ElementFromNode(call.ToExprAt(), &graph.OutputNode{
		Node: op,
		Shape: &shape.Shape{
			DType:       xShapes[0].DType,
			AxisLengths: op.(interface{ PJRTDims() []int }).PJRTDims(),
		},
	})
}

func evalLen(ctx interp.Context, call elements.CallAt, _ *elements.Func, _ *ir.FuncBuiltin, args []state.Element) (output state.Element, err error) {
	shape, err := elements.ShapeFromElement(args[0])
	if err != nil {
		return nil, err
	}
	length := ir.Int(shape.OuterAxisLength())
	value := values.AtomIntegerValue(call.Node().Type(), length)
	return ctx.Evaluator().ElementFromValue(elements.NewNodeAt[ir.Node](ctx.File(), call.Node()), value)
}

func evalSplit(ctx interp.Context, call elements.CallAt, fn *elements.Func, irFunc *ir.FuncBuiltin, args []state.Element) (state.Element, error) {
	node, firstArgShape, err := state.NodeFromElement(args[1])
	if err != nil {
		return nil, err
	}
	axis, err := elements.ConstantScalarFromElement[ir.Int](args[0])
	if err != nil {
		return nil, err
	}
	numSplits, err := elements.ConstantScalarFromElement[ir.Int](args[2])
	if err != nil {
		return nil, err
	}
	op, err := pjrtGraph(ctx).NewSplit(node, int(axis), int(numSplits))
	if err != nil {
		return nil, err
	}
	return ctx.State().ElementFromNode(call.ToExprAt(), &graph.OutputNode{
		Node: op,
		Shape: &shape.Shape{
			DType:       firstArgShape.DType,
			AxisLengths: op.(interface{ PJRTDims() []int }).PJRTDims(),
		},
	})
}

func evalExpand(ctx interp.Context, call elements.CallAt, fn *elements.Func, irFunc *ir.FuncBuiltin, args []state.Element) (state.Element, error) {
	targetAxes, err := elements.AxesFromElement(args[1])
	if err != nil {
		return nil, err
	}
	expandAxes := make([]int, len(targetAxes))
	for i := range targetAxes {
		expandAxes[i] = i
	}
	x, xShape, err := state.NodeFromElement(args[0])
	if err != nil {
		return nil, err
	}
	targetShape := &shape.Shape{
		DType:       xShape.DType,
		AxisLengths: targetAxes,
	}
	op, err := pjrtGraph(ctx).NewBroadcastInDim(x, targetShape, expandAxes)
	if err != nil {
		return nil, err
	}
	return ctx.State().ElementFromNode(call.ToExprAt(), &graph.OutputNode{
		Node:  op,
		Shape: targetShape,
	})
}

func evalGather(ctx interp.Context, call elements.CallAt, fn *elements.Func, irFunc *ir.FuncBuiltin, args []state.Element) (state.Element, error) {
	inputShape, err := elements.ShapeFromElement(args[0])
	if err != nil {
		return nil, err
	}
	indicesShape, err := elements.ShapeFromElement(args[1])
	if err != nil {
		return nil, err
	}
	paramsRank := len(inputShape.AxisLengths)
	indicesRank := len(indicesShape.AxisLengths)
	indexedSubRank := indicesShape.AxisLengths[indicesRank-1] // N from documentation.
	slicesSubRank := paramsRank - indexedSubRank              // S from documentation, the slices dimensions.
	if slicesSubRank < 0 {
		return nil, fmt.Errorf("Gather params are \"over-indexed\": params has only rank %d and "+
			"indexed rank is %d (last dimension of indices)", paramsRank, indexedSubRank)
	}
	outputSubRank := indicesRank - 1

	// * indexVectorDim is always the last one.
	indexVectorDim := indicesRank - 1
	// * startIndexMap is sequential and sorted
	startIndexMap := make([]int, indexedSubRank)
	for ii := 0; ii < indexedSubRank; ii++ {
		startIndexMap[ii] = ii
	}
	// * sliceSizes are 1 everywhere but on the sliced dimensions.
	// * collapsedSliceDims is set to collapse all dimensions set to 1.
	sliceSizes := make([]int, paramsRank)
	collapsedSliceDims := make([]int, indexedSubRank)
	for ii := 0; ii < paramsRank; ii++ {
		if ii < indexedSubRank {
			sliceSizes[ii] = 1
			collapsedSliceDims[ii] = ii
		} else {
			sliceSizes[ii] = inputShape.AxisLengths[ii]
		}
	}
	// * offsetDims are the dimensions indexed.
	offsetDims := make([]int, paramsRank-indexedSubRank)
	for ii := range offsetDims {
		offsetDims[ii] = outputSubRank + ii
	}

	x, xShape, err := state.NodeFromElement(args[0])
	if err != nil {
		return nil, err
	}
	indicesNode, _, err := state.NodeFromElement(args[1])
	if err != nil {
		return nil, err
	}
	op, err := pjrtGraph(ctx).NewGather(x, indicesNode, indexVectorDim, offsetDims, collapsedSliceDims, startIndexMap, sliceSizes, false)
	if err != nil {
		return nil, err
	}
	return ctx.State().ElementFromNode(call.ToExprAt(), &graph.OutputNode{
		Node: op,
		Shape: &shape.Shape{
			DType:       xShape.DType,
			AxisLengths: op.(interface{ PJRTDims() []int }).PJRTDims(),
		},
	})
}
