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
	"github.com/gx-org/gx/interp/grapheval"
	"github.com/gx-org/gx/interp"
	"github.com/gx-org/gx/interp/materialise"
)

func toStruct(ctx evaluator.Context, exprAt elements.ExprAt, tpl ops.Tuple, shapes []*shape.Shape, structTyp *ir.StructType) (*interp.Struct, error) {
	// Construct dummy expressions for all the fields of the structure to keep track of the value types.
	fieldExprs := make([]ir.AssignableExpr, structTyp.NumFields())
	for i, field := range structTyp.Fields.Fields() {
		fieldExprs[i] = &ir.ValueRef{
			Src:  field.Name,
			Stor: field.Storage(),
		}
	}
	ev := ctx.Evaluator().(*grapheval.Evaluator)
	els, err := ev.ElementsFromTupleNode(exprAt.File(), tpl, fieldExprs, shapes)
	if err != nil {
		return nil, err
	}
	return interp.NewStructFromElements(structTyp, els), nil
}

func packXLATuple(ctx evaluator.Context, elts []ir.Element) (ops.OutputNode, error) {
	outputNodes, _, err := materialise.Flatten(ctx.Materialiser(), elts...)
	if err != nil {
		return ops.OutputNode{}, err
	}
	tupleNode, err := ctx.Evaluator().ArrayOps().Graph().Core().Tuple(outputNodes)
	if err != nil {
		return ops.OutputNode{}, err
	}
	return ops.OutputNode{Node: tupleNode, Shape: &shape.Shape{DType: dtype.Invalid}}, nil
}

func evalWhile(ctx evaluator.Context, call elements.CallAt, fn interp.Func, irFunc *ir.FuncBuiltin, args []ir.Element) ([]ir.Element, error) {
	g := pjrtGraph(ctx)

	stateStruct := interp.Underlying(args[0]).(*interp.Struct)
	stateNodes, stateShapes, err := materialise.Flatten(ctx.Materialiser(), stateStruct)
	if err != nil {
		return nil, err
	}

	cond, err := grapheval.GraphFromElement("while.cond", args[1])
	if err != nil {
		return nil, err
	}
	body, err := grapheval.GraphFromElement("while.body", args[2])
	if err != nil {
		return nil, err
	}

	stateTpl, err := g.Tuple(stateNodes)
	if err != nil {
		return nil, err
	}
	fnState, err := g.While(cond, body, stateTpl)
	if err != nil {
		return nil, err
	}
	out, err := toStruct(ctx, call.ToExprAt(), fnState.(ops.Tuple), stateShapes, stateStruct.StructType())
	if err != nil {
		return nil, err
	}
	return []ir.Element{out}, nil
}
