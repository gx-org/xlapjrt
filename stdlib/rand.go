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
	"github.com/gx-org/backend/graph"
	"github.com/gx-org/backend/shape"
	"github.com/gx-org/gx/build/ir"
	"github.com/gx-org/gx/interp/elements"
	"github.com/gx-org/gx/interp"
	"github.com/gx-org/gx/interp/state"
	xlagraph "github.com/gx-org/xlapjrt/backend/graph"
)

var philoxStateShape = &shape.Shape{
	DType:       dtype.Uint64,
	AxisLengths: []int{3},
}

func evalPhilox(ctx interp.Context, call elements.CallAt, fn *elements.Func, irFunc *ir.FuncBuiltin, args []state.Element, dtyp dtype.DataType) (output state.Element, err error) {
	philox := fn.Recv().Element.(elements.FieldSelector)
	exprAt := call.ToExprAt()
	field, err := philox.SelectField(exprAt, "state")
	if err != nil {
		return nil, err
	}
	stateNode, _, err := state.NodeFromElement(field)
	if err != nil {
		return nil, err
	}
	dimensions, err := elements.AxesFromElement(args[0])
	if err != nil {
		return nil, err
	}
	g := ctx.State()
	bckGraph := g.BackendGraph().(*xlagraph.Graph)
	targetShape := &shape.Shape{DType: dtyp, AxisLengths: dimensions}
	newState, values, err := bckGraph.NewRngBitGenerator(stateNode, targetShape)
	if err != nil {
		return nil, err
	}

	philoxState := call.Node().ExprFromResult(0)
	philoxStateAt := elements.NewNodeAt[ir.Expr](call.File(), philoxState)
	philoxStruct := ir.Underlying(philoxState.Type()).(*ir.StructType)
	stateArray := philoxStruct.Fields.Fields()[0]
	stateArrayAt := elements.NewNodeAt[ir.Expr](call.File(), stateArray)
	philoxStateElement, err := ctx.State().ElementFromNode(stateArrayAt, &graph.OutputNode{
		Node:  newState,
		Shape: philoxStateShape,
	})
	if err != nil {
		return nil, err
	}
	randValueAt := elements.NewNodeAt[ir.Expr](call.File(), call.Node().ExprFromResult(1))
	valuesElement, err := ctx.State().ElementFromNode(randValueAt, &graph.OutputNode{
		Node:  values,
		Shape: targetShape,
	})
	if err != nil {
		return nil, err
	}
	return elements.NewTuple(call.File(), call.Node(), []elements.Element{
		elements.NewMethods(elements.NewStruct(
			philoxStruct,
			philoxStateAt,
			map[string]state.Element{"state": philoxStateElement},
		)),
		valuesElement,
	}), nil
}

func evalPhiloxUint32(ctx interp.Context, call elements.CallAt, fn *elements.Func, irFunc *ir.FuncBuiltin, args []state.Element) (output state.Element, err error) {
	return evalPhilox(ctx, call, fn, irFunc, args, dtype.Uint32)
}

func evalPhiloxUint64(ctx interp.Context, call elements.CallAt, fn *elements.Func, irFunc *ir.FuncBuiltin, args []state.Element) (output state.Element, err error) {
	return evalPhilox(ctx, call, fn, irFunc, args, dtype.Uint64)
}
