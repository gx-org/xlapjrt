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
	"slices"

	"github.com/gomlx/gopjrt/xlabuilder"
	"github.com/gx-org/backend/ops"
	"github.com/gx-org/backend/shape"
	"github.com/gx-org/gx/build/ir"
	"github.com/gx-org/gx/interp/elements"
	"github.com/gx-org/gx/interp/evaluator"
	"github.com/gx-org/gx/interp/grapheval"
	"github.com/gx-org/gx/interp"
)

func xlaReductionFunc(f func(*xlabuilder.Op, ...int) (*xlabuilder.Op, error)) interp.FuncBuiltin {
	return func(ctx evaluator.Context, call elements.CallAt, fn elements.Func, irFunc *ir.FuncBuiltin, args []elements.Element) ([]elements.Element, error) {
		x, xShape, err := grapheval.NodeFromElement(ctx.Evaluator().ArrayOps(), args[0])
		if err != nil {
			return nil, err
		}
		axes, err := elements.AxesFromElement(args[1])
		if err != nil {
			return nil, err
		}
		if len(axes) == 0 {
			// Note that we diverge from XLA's behavior: if no reduction axes are
			// specified, treat this as a no-op.
			return []elements.Element{args[0]}, nil
		}
		resultNode, err := pjrtGraph(ctx).ReduceFunc(x, axes, f)
		if err != nil {
			return nil, err
		}
		return grapheval.ElementsFromNode(call.ToExprAt(), &ops.OutputNode{
			Node: resultNode,
			Shape: &shape.Shape{
				DType:       xShape.DType,
				AxisLengths: resultNode.(interface{ PJRTDims() []int }).PJRTDims(),
			},
		})
	}
}

func evalTranspose(ctx evaluator.Context, call elements.CallAt, fn elements.Func, irFunc *ir.FuncBuiltin, args []elements.Element) ([]elements.Element, error) {
	argNode, argShape, err := grapheval.NodeFromElement(ctx.Evaluator().ArrayOps(), args[0])
	if err != nil {
		return nil, err
	}
	if len(argShape.AxisLengths) <= 1 {
		return []elements.Element{args[0]}, nil
	}
	wantAxes := make([]int, len(argShape.AxisLengths))
	for i := range wantAxes {
		wantAxes[i] = len(wantAxes) - i - 1
	}
	op, err := pjrtGraph(ctx).Transpose(argNode, wantAxes)
	if err != nil {
		return nil, err
	}
	targetLengths := append([]int{}, argShape.AxisLengths...)
	slices.Reverse(targetLengths)
	targetShape := &shape.Shape{
		DType:       argShape.DType,
		AxisLengths: targetLengths,
	}
	return grapheval.ElementsFromNode(call.ToExprAt(), &ops.OutputNode{
		Node:  op,
		Shape: targetShape,
	})
}

func evalEinsum(ctx evaluator.Context, call elements.CallAt, fn elements.Func, irFunc *ir.FuncBuiltin, args []elements.Element) ([]elements.Element, error) {
	evaluator := ctx.Evaluator()
	left, leftShape, err := grapheval.NodeFromElement(evaluator.ArrayOps(), args[0])
	if err != nil {
		return nil, err
	}
	lhsContractingAxes, err := elements.AxesFromElement(args[1])
	if err != nil {
		return nil, err
	}
	lhsBatchAxes, err := elements.AxesFromElement(args[2])
	if err != nil {
		return nil, err
	}
	right, rightShape, err := grapheval.NodeFromElement(evaluator.ArrayOps(), args[3])
	if err != nil {
		return nil, err
	}
	rhsContractingAxes, err := elements.AxesFromElement(args[4])
	if err != nil {
		return nil, err
	}
	rhsBatchAxes, err := elements.AxesFromElement(args[5])
	if err != nil {
		return nil, err
	}

	op, err := pjrtGraph(ctx).DotGeneral(left, right,
		[2][]int{lhsBatchAxes, rhsBatchAxes},
		[2][]int{lhsContractingAxes, rhsContractingAxes})
	if err != nil {
		return nil, fmt.Errorf("\nlhsContractingAxes: %v\nlhsBatchAxes: %v\nrhsContractingAxes: %v\nrhsBatchAxes: %v\nleft: %v\nright: %v", lhsContractingAxes, lhsBatchAxes, rhsContractingAxes, rhsBatchAxes, leftShape, rightShape)
	}
	return grapheval.ElementsFromNode(call.ToExprAt(), &ops.OutputNode{
		Node: op,
		Shape: &shape.Shape{
			DType:       leftShape.DType,
			AxisLengths: op.(interface{ PJRTDims() []int }).PJRTDims(),
		},
	})
}

func evalIota(ctx evaluator.Context, call elements.CallAt, fn elements.Func, irFunc *ir.FuncBuiltin, args []elements.Element) ([]elements.Element, error) {
	axes, err := elements.AxesFromElement(args[0])
	if err != nil {
		return nil, err
	}
	axisIndex, err := elements.ConstantIntFromElement(args[1])
	if err != nil {
		return nil, err
	}
	targetShape := &shape.Shape{
		DType:       ir.DefaultIntKind.DType(),
		AxisLengths: axes,
	}
	op, err := pjrtGraph(ctx).Iota(targetShape, axisIndex)
	if err != nil {
		return nil, err
	}
	return grapheval.ElementsFromNode(call.ToExprAt(), &ops.OutputNode{
		Node:  op,
		Shape: targetShape,
	})
}

func evalArgmax(ctx evaluator.Context, call elements.CallAt, fn elements.Func, irFunc *ir.FuncBuiltin, args []elements.Element) ([]elements.Element, error) {
	argNode, _, err := grapheval.NodeFromElement(ctx.Evaluator().ArrayOps(), args[0])
	if err != nil {
		return nil, err
	}
	axisIndex, err := elements.ConstantScalarFromElement[ir.Int](args[1])
	if err != nil {
		return nil, err
	}
	op, err := pjrtGraph(ctx).ArgMinMax(argNode, int(axisIndex), ir.DefaultIntKind, false)
	if err != nil {
		return nil, err
	}
	return grapheval.ElementsFromNode(call.ToExprAt(), &ops.OutputNode{
		Node: op,
		Shape: &shape.Shape{
			DType:       ir.DefaultIntKind.DType(),
			AxisLengths: op.(interface{ PJRTDims() []int }).PJRTDims(),
		},
	})
}
