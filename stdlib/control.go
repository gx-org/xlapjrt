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
	pjrtgraph "github.com/gx-org/xlapjrt/backend/graph"
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
	els, err := grapheval.ElementsFromTupleNode(ctx.Evaluator().ArrayOps().Graph(), exprAt.File(), exprAt.Node(), tpl, fieldExprs, shapes)
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

func buildSubgraph(ectx evaluator.Context, call elements.CallAt, fn ir.Func, tupleShapes []*shape.Shape, structTyp *ir.StructType, resultHandler func(evaluator.Context, []ir.Element) (ops.OutputNode, error)) (*ops.Subgraph, error) {
	ctx := ectx.(*interp.FileScope)
	g := pjrtGraph(ctx)
	subgraph, err := g.Core().Subgraph(fn.Name())
	if err != nil {
		return nil, err
	}
	core := subgraph.Core().(*pjrtgraph.Graph)
	fnState, err := core.TupleArgument("_", 0, tupleShapes)
	if err != nil {
		return nil, err
	}

	stateStruct, err := toStruct(ctx, call.ToExprAt(), fnState, tupleShapes, structTyp)
	if err != nil {
		return nil, err
	}
	evaluator := ctx.Evaluator().(*grapheval.Evaluator)
	subeval := grapheval.New(evaluator.Importer(), nil, subgraph)
	resultElt, err := ctx.EvalFunctionToElement(subeval, fn, []ir.Element{stateStruct})
	if err != nil {
		return nil, err
	}
	output, err := resultHandler(ectx, resultElt)
	if err != nil {
		return nil, err
	}
	return &ops.Subgraph{Graph: subgraph, Result: output}, nil
}

func evalWhile(ctx evaluator.Context, call elements.CallAt, fn interp.Func, irFunc *ir.FuncBuiltin, args []ir.Element) ([]ir.Element, error) {
	g := pjrtGraph(ctx)

	stateStruct := (args[0]).(*interp.Struct)
	stateNodes, stateShapes, err := toNodes(ctx, stateStruct)
	if err != nil {
		return nil, err
	}

	cond := args[1].(interp.Func)
	body := args[2].(interp.Func)
	condSG, err := buildSubgraph(ctx, call, cond.Func(), stateShapes, stateStruct.StructType(), getOutputNode)
	if err != nil {
		return nil, err
	}
	bodySG, err := buildSubgraph(ctx, call, body.Func(), stateShapes, stateStruct.StructType(), packXLATuple)
	if err != nil {
		return nil, err
	}

	stateTpl, err := g.Tuple(stateNodes)
	if err != nil {
		return nil, err
	}
	fnState, err := g.While(*condSG, *bodySG, stateTpl)
	if err != nil {
		return nil, err
	}
	out, err := toStruct(ctx, call.ToExprAt(), fnState.(ops.Tuple), stateShapes, stateStruct.StructType())
	if err != nil {
		return nil, err
	}
	return []ir.Element{out}, nil
}
