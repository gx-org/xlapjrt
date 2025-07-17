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
	"github.com/pkg/errors"
	"github.com/gx-org/backend/dtype"
	"github.com/gx-org/backend/ops"
	"github.com/gx-org/backend/shape"
	"github.com/gx-org/gx/build/ir"
	"github.com/gx-org/gx/internal/interp/flatten"
	"github.com/gx-org/gx/interp/elements"
	"github.com/gx-org/gx/interp/evaluator"
	"github.com/gx-org/gx/interp/grapheval"
	"github.com/gx-org/gx/interp"
)

func toNodes(ctx ir.Evaluator, elts ...ir.Element) ([]ops.Node, []*shape.Shape, error) {
	elts, err := flatten.Flatten(elts...)
	if err != nil {
		return nil, nil, err
	}
	outputs, err := grapheval.MaterialiseAll(ctx, elts)
	if err != nil {
		return nil, nil, err
	}
	result := make([]ops.Node, len(outputs))
	shapes := make([]*shape.Shape, len(outputs))
	for i, output := range outputs {
		result[i] = output.Node
		shapes[i] = output.Shape
	}
	return result, shapes, nil
}

func toStruct(ctx evaluator.Context, exprAt elements.ExprAt, tpl ops.Tuple, shapes []*shape.Shape, structTyp *ir.StructType) (*interp.Struct, error) {
	// Construct dummy expressions for all the fields of the structure to keep track of the value types.
	fieldExprs := make([]ir.AssignableExpr, structTyp.NumFields())
	for i, field := range structTyp.Fields.Fields() {
		fieldExprs[i] = &ir.ValueRef{
			Src:  field.Name,
			Stor: field.Storage(),
		}
	}
	els, err := grapheval.ElementsFromTupleNode(exprAt.File(), tpl, fieldExprs, shapes)
	if err != nil {
		return nil, err
	}
	return interp.NewStructFromElements(structTyp, els), nil
}

func getOutputNode(ctx evaluator.Context, elts []ir.Element) (ops.OutputNode, error) {
	if len(elts) != 1 {
		return ops.OutputNode{}, errors.Errorf("cannot get the output node of %d element(s)", len(elts))
	}
	node, shape, err := grapheval.NodeFromElement(ctx, elts[0])
	return ops.OutputNode{Node: node, Shape: shape}, err
}

func packXLATuple(ctx evaluator.Context, elts []ir.Element) (ops.OutputNode, error) {
	outputNodes, _, err := toNodes(ctx, elts...)
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
	stateNodes, stateShapes, err := toNodes(ctx, stateStruct)
	if err != nil {
		return nil, err
	}

	cond, err := grapheval.GraphFromElement(args[1])
	if err != nil {
		return nil, err
	}
	body, err := grapheval.GraphFromElement(args[2])
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
