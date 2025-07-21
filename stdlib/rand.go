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
	"github.com/gx-org/backend/ops"
	"github.com/gx-org/backend/shape"
	"github.com/gx-org/gx/build/ir"
	"github.com/gx-org/gx/interp/elements"
	"github.com/gx-org/gx/interp/evaluator"
	"github.com/gx-org/gx/interp"
	"github.com/gx-org/gx/interp/materialise"
	xlagraph "github.com/gx-org/xlapjrt/backend/graph"
)

var philoxStateShape = &shape.Shape{
	DType:       dtype.Uint64,
	AxisLengths: []int{3},
}

func evalPhilox(ctx evaluator.Context, call elements.CallAt, fn interp.Func, irFunc *ir.FuncBuiltin, args []ir.Element, dtyp dtype.DataType) ([]ir.Element, error) {
	fitp := ctx.(*interp.FileScope)
	philox := fn.Recv().Element
	philoxStruct := ir.Underlying(philox.NamedType()).(*ir.StructType)
	stateArray := philoxStruct.Fields.FindField("state")
	field, err := philox.Select(&ir.SelectorExpr{
		X:    call.Node(),
		Stor: stateArray.Storage(),
	})
	if err != nil {
		return nil, err
	}
	evaluator := ctx.Evaluator()
	stateNode, _, err := materialise.Element(ctx.Materialiser(), field)
	if err != nil {
		return nil, err
	}
	dimensions, err := elements.AxesFromElement(args[0])
	if err != nil {
		return nil, err
	}
	bckGraph := evaluator.ArrayOps().Graph().(*xlagraph.Graph)
	targetShape := &shape.Shape{DType: dtyp, AxisLengths: dimensions}
	newState, values, err := bckGraph.RngBitGenerator(stateNode, targetShape)
	if err != nil {
		return nil, err
	}

	philoxStateElement, err := ctx.Materialiser().ElementsFromNodes(
		call.File(),
		&ir.ValueRef{
			Src:  stateArray.Name,
			Stor: stateArray.Storage(),
		},
		&ops.OutputNode{
			Node:  newState,
			Shape: philoxStateShape,
		})
	if err != nil {
		return nil, err
	}
	valuesElement, err := ctx.Materialiser().ElementsFromNodes(
		call.File(),
		call.Node().ExprFromResult(1),
		&ops.OutputNode{
			Node:  values,
			Shape: targetShape,
		})
	if err != nil {
		return nil, err
	}
	return []ir.Element{
		interp.NewNamedType(fitp.NewFunc, philox.NamedType(), interp.NewStruct(
			philoxStruct,
			map[string]ir.Element{"state": philoxStateElement[0]},
		)),
		valuesElement[0],
	}, nil
}

func evalPhiloxUint32(ctx evaluator.Context, call elements.CallAt, fn interp.Func, irFunc *ir.FuncBuiltin, args []ir.Element) ([]ir.Element, error) {
	return evalPhilox(ctx, call, fn, irFunc, args, dtype.Uint32)
}

func evalPhiloxUint64(ctx evaluator.Context, call elements.CallAt, fn interp.Func, irFunc *ir.FuncBuiltin, args []ir.Element) ([]ir.Element, error) {
	return evalPhilox(ctx, call, fn, irFunc, args, dtype.Uint64)
}
