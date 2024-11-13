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
	"github.com/gx-org/backend/shape"
	"github.com/gx-org/gx/build/ir"
	"github.com/gx-org/gx/interp"
	"github.com/gx-org/gx/interp/state"
)

func xlaReductionFunc(f func(*xlabuilder.Op, ...int) (*xlabuilder.Op, error)) interp.FuncBuiltin {
	return func(ctx interp.Context, call state.CallAt, fn *state.Func, irFunc *ir.FuncBuiltin, args []state.Element) (state.Element, error) {
		x, xShape, err := state.NodeFromElement(args[0])
		if err != nil {
			return nil, err
		}
		axes, err := state.AxesFromElement(args[1])
		if err != nil {
			return nil, err
		}
		if len(axes) == 0 {
			// Note that we diverge from XLA's behavior: if no reduction axes are
			// specified, treat this as a no-op.
			return args[0], nil
		}
		resultNode, err := pjrtGraph(ctx).NewReduceFunc(x, axes, f)
		if err != nil {
			return nil, err
		}
		return ctx.State().ElementFromNode(call.ToExprAt(), resultNode, &shape.Shape{
			DType:       xShape.DType,
			AxisLengths: resultNode.(interface{ PJRTDims() []int }).PJRTDims(),
		})
	}
}

func evalTranspose(ctx interp.Context, call state.CallAt, fn *state.Func, irFunc *ir.FuncBuiltin, args []state.Element) (state.Element, error) {
	argNode, argShape, err := state.NodeFromElement(args[0])
	if err != nil {
		return nil, err
	}
	if len(argShape.AxisLengths) <= 1 {
		return args[0], nil
	}
	wantAxes := make([]int, len(argShape.AxisLengths))
	for i := range wantAxes {
		wantAxes[i] = len(wantAxes) - i - 1
	}
	op, err := pjrtGraph(ctx).NewTranspose(argNode, wantAxes)
	if err != nil {
		return nil, err
	}
	targetLengths := append([]int{}, argShape.AxisLengths...)
	slices.Reverse(targetLengths)
	targetShape := &shape.Shape{
		DType:       argShape.DType,
		AxisLengths: targetLengths,
	}
	return ctx.State().ElementFromNode(call.ToExprAt(), op, targetShape)
}

func evalEinsum(ctx interp.Context, call state.CallAt, fn *state.Func, irFunc *ir.FuncBuiltin, args []state.Element) (state.Element, error) {
	left, leftShape, err := state.NodeFromElement(args[0])
	if err != nil {
		return nil, err
	}
	lhsContractingAxes, err := state.AxesFromElement(args[1])
	if err != nil {
		return nil, err
	}
	lhsBatchAxes, err := state.AxesFromElement(args[2])
	if err != nil {
		return nil, err
	}
	right, rightShape, err := state.NodeFromElement(args[3])
	if err != nil {
		return nil, err
	}
	rhsContractingAxes, err := state.AxesFromElement(args[4])
	if err != nil {
		return nil, err
	}
	rhsBatchAxes, err := state.AxesFromElement(args[5])
	if err != nil {
		return nil, err
	}

	op, err := pjrtGraph(ctx).NewDotGeneral(left, right,
		[2][]int{lhsBatchAxes, rhsBatchAxes},
		[2][]int{lhsContractingAxes, rhsContractingAxes})
	if err != nil {
		return nil, fmt.Errorf("\nlhsContractingAxes: %v\nlhsBatchAxes: %v\nrhsContractingAxes: %v\nrhsBatchAxes: %v\nleft: %v\nright: %v", lhsContractingAxes, lhsBatchAxes, rhsContractingAxes, rhsBatchAxes, leftShape, rightShape)
	}
	return ctx.State().ElementFromNode(call.ToExprAt(), op, &shape.Shape{
		DType:       leftShape.DType,
		AxisLengths: op.(interface{ PJRTDims() []int }).PJRTDims(),
	})
}

func evalIota(ctx interp.Context, call state.CallAt, fn *state.Func, irFunc *ir.FuncBuiltin, args []state.Element) (state.Element, error) {
	axes, err := state.AxesFromElement(args[0])
	if err != nil {
		return nil, err
	}
	axisIndex, err := state.ConstantScalarFromElement[ir.Int](args[1])
	if err != nil {
		return nil, err
	}
	targetShape := &shape.Shape{
		DType:       ir.DefaultIntKind.DType(),
		AxisLengths: axes,
	}
	op, err := pjrtGraph(ctx).NewIota(targetShape, int(axisIndex))
	if err != nil {
		return nil, err
	}
	return ctx.State().ElementFromNode(call.ToExprAt(), op, targetShape)
}

func evalIotaFull(ctx interp.Context, call state.CallAt, fn *state.Func, irFunc *ir.FuncBuiltin, args []state.Element) (state.Element, error) {
	axes, err := state.AxesFromElement(args[0])
	if err != nil {
		return nil, err
	}
	targetShape := &shape.Shape{
		DType:       ir.DefaultIntKind.DType(),
		AxisLengths: axes,
	}
	iotaOp, err := pjrtGraph(ctx).NewIota(&shape.Shape{
		DType:       ir.DefaultIntKind.DType(),
		AxisLengths: []int{targetShape.Size()},
	}, 0)
	if err != nil {
		return nil, err
	}
	op, err := pjrtGraph(ctx).NewReshape(iotaOp, targetShape.AxisLengths)
	if err != nil {
		return nil, err
	}
	return ctx.State().ElementFromNode(call.ToExprAt(), op, targetShape)
}

func evalArgmax(ctx interp.Context, call state.CallAt, fn *state.Func, irFunc *ir.FuncBuiltin, args []state.Element) (state.Element, error) {
	argNode, _, err := state.NodeFromElement(args[0])
	if err != nil {
		return nil, err
	}
	axisIndex, err := state.ConstantScalarFromElement[ir.Int](args[1])
	if err != nil {
		return nil, err
	}
	op, err := pjrtGraph(ctx).NewArgMinMax(argNode, int(axisIndex), ir.DefaultIntKind, false)
	if err != nil {
		return nil, err
	}
	return ctx.State().ElementFromNode(call.ToExprAt(), op, &shape.Shape{
		DType:       ir.DefaultIntKind.DType(),
		AxisLengths: op.(interface{ PJRTDims() []int }).PJRTDims(),
	})
}
