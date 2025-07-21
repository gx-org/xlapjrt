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
	"github.com/gx-org/backend/ops"
	"github.com/gx-org/gx/build/ir"
	"github.com/gx-org/gx/interp"
	"github.com/gx-org/gx/interp/elements"
	"github.com/gx-org/gx/interp/evaluator"
	"github.com/gx-org/gx/interp/grapheval"
	"github.com/gx-org/gx/interp/materialise"
)

func evalWhile(ctx evaluator.Context, call elements.CallAt, fn interp.Func, irFunc *ir.FuncBuiltin, args []ir.Element) ([]ir.Element, error) {
	g := pjrtGraph(ctx)

	cond, err := grapheval.GraphFromElement("while.cond", args[1])
	if err != nil {
		return nil, err
	}
	body, err := grapheval.GraphFromElement("while.body", args[2])
	if err != nil {
		return nil, err
	}

	stateNodes, stateShapes, err := materialise.Flatten(ctx.Materialiser(), args[0])
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
	ev := ctx.Evaluator().(*grapheval.Evaluator)
	out, err := ev.ElementFromTuple(ctx.File(), call.Node(), fnState.(ops.Tuple), stateShapes, args[0].Type())
	if err != nil {
		return nil, err
	}
	return []ir.Element{out}, nil
}
