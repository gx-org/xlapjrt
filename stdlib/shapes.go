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
	"github.com/gx-org/gx/interp/evaluator"
	"github.com/gx-org/gx/interp/grapheval"
)

func evalConcat(ctx evaluator.Context, call elements.CallAt, fn elements.Func, irFunc *ir.FuncBuiltin, args []elements.Element) ([]elements.Element, error) {
	xs := make([]graph.Node, len(args)-1)
	xShapes := make([]*shape.Shape, len(args)-1)
	ao := ctx.Evaluation().Evaluator().ArrayOps()
	for i, arg := range args[1:] {
		var err error
		xs[i], xShapes[i], err = grapheval.NodeFromElement(ao, arg)
		if err != nil {
			return nil, err
		}
	}
	axis, err := elements.ConstantScalarFromElement[ir.Int](args[0])
	if err != nil {
		return nil, err
	}
	op, err := pjrtGraph(ctx).Concat(int(axis), xs)
	if err != nil {
		return nil, err
	}
	return grapheval.ElementsFromNode(call.ToExprAt(), &graph.OutputNode{
		Node: op,
		Shape: &shape.Shape{
			DType:       xShapes[0].DType,
			AxisLengths: op.(interface{ PJRTDims() []int }).PJRTDims(),
		},
	})
}

func evalLen(ctx evaluator.Context, call elements.CallAt, _ elements.Func, _ *ir.FuncBuiltin, args []elements.Element) ([]elements.Element, error) {
	shape, err := elements.ShapeFromElement(args[0])
	if err != nil {
		return nil, err
	}
	length := ir.Int(shape.OuterAxisLength())
	value, err := values.AtomIntegerValue(call.Node().Type(), length)
	if err != nil {
		return nil, err
	}
	out, err := ctx.Evaluation().Evaluator().ElementFromAtom(elements.NewExprAt(ctx.File(), call.Node()), value)
	if err != nil {
		return nil, err
	}
	return []elements.Element{out}, nil
}

func evalSplit(ctx evaluator.Context, call elements.CallAt, fn elements.Func, irFunc *ir.FuncBuiltin, args []elements.Element) ([]elements.Element, error) {
	ao := ctx.Evaluation().Evaluator().ArrayOps()
	node, firstArgShape, err := grapheval.NodeFromElement(ao, args[1])
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
	op, err := pjrtGraph(ctx).Split(node, int(axis), int(numSplits))
	if err != nil {
		return nil, err
	}
	return grapheval.ElementsFromNode(call.ToExprAt(), &graph.OutputNode{
		Node: op,
		Shape: &shape.Shape{
			DType:       firstArgShape.DType,
			AxisLengths: op.(interface{ PJRTDims() []int }).PJRTDims(),
		},
	})
}

func evalGather(ctx evaluator.Context, call elements.CallAt, fn elements.Func, irFunc *ir.FuncBuiltin, args []elements.Element) ([]elements.Element, error) {
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

	ao := ctx.Evaluation().Evaluator().ArrayOps()
	x, xShape, err := grapheval.NodeFromElement(ao, args[0])
	if err != nil {
		return nil, err
	}
	indicesNode, _, err := grapheval.NodeFromElement(ao, args[1])
	if err != nil {
		return nil, err
	}
	op, err := pjrtGraph(ctx).Gather(x, indicesNode, indexVectorDim, offsetDims, collapsedSliceDims, startIndexMap, sliceSizes, false)
	if err != nil {
		return nil, err
	}
	return grapheval.ElementsFromNode(call.ToExprAt(), &graph.OutputNode{
		Node: op,
		Shape: &shape.Shape{
			DType:       xShape.DType,
			AxisLengths: op.(interface{ PJRTDims() []int }).PJRTDims(),
		},
	})
}
