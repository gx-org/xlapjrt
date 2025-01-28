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
	pjrtgraph "github.com/gx-org/xlapjrt/backend/graph"
)

func toNodes(elt state.Element) ([]graph.Node, []*shape.Shape, error) {
	elts, err := elt.Flatten()
	if err != nil {
		return nil, nil, err
	}
	outputs, err := state.MaterialiseAll(elts)
	if err != nil {
		return nil, nil, err
	}
	result := make([]graph.Node, len(outputs))
	shapes := make([]*shape.Shape, len(outputs))
	for i, output := range outputs {
		result[i] = output.Node
		shapes[i] = output.Shape
	}
	return result, shapes, nil
}

func toStruct(ctx interp.Context, exprAt elements.ExprAt, tpl graph.Tuple, shapes []*shape.Shape, structTyp *ir.StructType) (*elements.Struct, error) {
	stateTpl, err := ctx.State().ElementFromTuple(exprAt, tpl, shapes)
	if err != nil {
		return nil, err
	}
	return elements.NewStructFromElements(structTyp, exprAt, stateTpl.Elements()), nil
}

func getOutputNode(_ *pjrtgraph.Graph, elt state.Element) (graph.OutputNode, error) {
	node, shape, err := state.NodeFromElement(elt)
	return graph.OutputNode{Node: node, Shape: shape}, err
}

func packXLATuple(g *pjrtgraph.Graph, elt state.Element) (graph.OutputNode, error) {
	outputNodes, _, err := toNodes(elt)
	if err != nil {
		return graph.OutputNode{}, err
	}

	tupleNode, err := g.NewTuple(outputNodes)
	if err != nil {
		return graph.OutputNode{}, err
	}
	return graph.OutputNode{Node: tupleNode, Shape: &shape.Shape{DType: dtype.Invalid}}, nil
}

func buildSubgraph(ctx interp.Context, call elements.CallAt, fn ir.Func, tupleShapes []*shape.Shape, structTyp *ir.StructType, resultHandler func(g *pjrtgraph.Graph, elt state.Element) (graph.OutputNode, error)) (*graph.Subgraph, error) {
	g := pjrtGraph(ctx)
	subgraph, err := g.Core().NewSubgraph(fn.Name())
	if err != nil {
		return nil, err
	}
	core := subgraph.Core().(*pjrtgraph.Graph)
	fnState, err := core.NewTupleArgument("_", 0, tupleShapes)
	if err != nil {
		return nil, err
	}

	stateStruct, err := toStruct(ctx, call.ToExprAt(), fnState.(graph.Tuple), tupleShapes, structTyp)
	if err != nil {
		return nil, err
	}

	resultElt, err := ctx.BuildGraph(fn, subgraph, []state.Element{stateStruct})
	if err != nil {
		return nil, err
	}
	output, err := resultHandler(g, resultElt)
	if err != nil {
		return nil, err
	}
	return &graph.Subgraph{Graph: subgraph, Result: output}, nil
}

func evalWhile(ctx interp.Context, call elements.CallAt, fn *elements.Func, irFunc *ir.FuncBuiltin, args []state.Element) (state.Element, error) {
	g := pjrtGraph(ctx)

	stateStruct := (args[0]).(*elements.Struct)
	stateNodes, stateShapes, err := toNodes(stateStruct)
	if err != nil {
		return nil, err
	}

	cond := args[1].(*elements.Func)
	body := args[2].(*elements.Func)
	condSG, err := buildSubgraph(ctx, call, cond.Func(), stateShapes, stateStruct.StructType(), getOutputNode)
	if err != nil {
		return nil, err
	}
	bodySG, err := buildSubgraph(ctx, call, body.Func(), stateShapes, stateStruct.StructType(), packXLATuple)
	if err != nil {
		return nil, err
	}

	stateTpl, err := g.NewTuple(stateNodes)
	if err != nil {
		return nil, err
	}
	fnState, err := g.NewWhile(*condSG, *bodySG, stateTpl)
	if err != nil {
		return nil, err
	}
	return toStruct(ctx, call.ToExprAt(), fnState.(graph.Tuple), stateShapes, stateStruct.StructType())
}
