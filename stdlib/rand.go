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
	"github.com/gx-org/backend/dtype"
	"github.com/gx-org/backend/shape"
	"github.com/gx-org/gx/build/ir"
	"github.com/gx-org/gx/interp"
	"github.com/gx-org/gx/interp/state"
	"github.com/gx-org/xlapjrt/backend/graph"
)

var philoxStateShape = &shape.Shape{
	DType:       dtype.Uint64,
	AxisLengths: []int{3},
}

func evalPhiloxUint64(ctx interp.Context, call state.CallAt, fn *state.Func, irFunc *ir.FuncBuiltin, args []state.Element) (output state.Element, err error) {
	philox := fn.Recv().Element.(state.FieldSelector)
	exprAt := call.ToExprAt()
	field, err := philox.SelectField(exprAt, 0)
	if err != nil {
		return nil, err
	}
	stateNode, _, err := state.NodeFromElement(field)
	if err != nil {
		return nil, err
	}
	dimensions, err := state.AxesFromElement(args[0])
	if err != nil {
		return nil, err
	}
	g := ctx.State()
	bckGraph := g.BackendGraph().(*graph.Graph)
	targetShape := &shape.Shape{DType: dtype.Uint64, AxisLengths: dimensions}
	newState, values, err := bckGraph.NewRngBitGenerator(stateNode, targetShape)
	if err != nil {
		return nil, err
	}

	philoxState := call.ExprT().ExprFromResult(0)
	philoxStateAt := state.NewExprAt[ir.Expr](call.File(), philoxState)
	philoxStruct := ir.Underlying(philoxState.Type()).(*ir.StructType)
	stateArray := philoxStruct.Fields.Fields()[0]
	stateArrayAt := state.NewExprAt[ir.Expr](call.File(), stateArray)
	philoxStateElement, err := ctx.State().ElementFromNode(stateArrayAt, newState, philoxStateShape)
	if err != nil {
		return nil, err
	}
	randValueAt := state.NewExprAt[ir.Expr](call.File(), call.ExprT().ExprFromResult(1))
	valuesElement, err := ctx.State().ElementFromNode(randValueAt, values, targetShape)
	if err != nil {
		return nil, err
	}
	return g.Tuple(call.File(), call.ExprT(), []state.Element{
		g.Methods(ctx.State().Struct(
			philoxStruct,
			philoxStateAt,
			[]state.Element{philoxStateElement},
		)),
		valuesElement,
	}), nil
}
