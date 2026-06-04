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

// Package stdlib provides the GX standard library for the PJRT (gopjrt) backend.
package stdlib

import (
	"fmt"

	"github.com/gomlx/gopjrt/xlabuilder"
	"github.com/gx-org/backend/ops"
	"github.com/gx-org/backend/shape"
	"github.com/gx-org/gx/build/ir"
	"github.com/gx-org/gx/interp/engine"
	"github.com/gx-org/gx/interp"
	"github.com/gx-org/gx/interp/materialise"
	"github.com/gx-org/gx/stdlib/builtin"
	"github.com/gx-org/gx/stdlib/impl"
	pjrtgraph "github.com/gx-org/xlapjrt/backend/graph"
)

// Stdlib is the PJRT implementation of the standard library.
var Stdlib = &impl.Stdlib{
	Num: impl.Num{
		Einsum: evalEinsum,
	},
	Rand: impl.Rand{
		PhiloxUint32: evalPhiloxUint32,
		PhiloxUint64: evalPhiloxUint64,
	},
	Shapes: impl.Shapes{
		Len: evalLen,
	},
}

func xlaUnaryFunc(f func(*xlabuilder.Op) (*xlabuilder.Op, error)) interp.FuncBuiltin {
	return func(env engine.Env, call *ir.FuncCallExpr, recv ir.Element, args []ir.Element) ([]ir.Element, error) {
		if len(args) != 1 {
			return nil, fmt.Errorf("unary function expects 1 argument, got %d", len(args))
		}
		mat := builtin.Materialiser(env)
		x, xShape, err := materialise.Element(mat, args[0])
		if err != nil {
			return nil, err
		}
		node, err := pjrtGraph(env).UnaryFunc(x, f)
		if err != nil {
			return nil, err
		}
		return materialise.ElementFromNode(env.File(), mat, &ops.OutputNode{
			Node:  node,
			Shape: xShape,
		}, call.Type())
	}
}

func xlaBinaryFunc(f func(x *xlabuilder.Op, y *xlabuilder.Op) (*xlabuilder.Op, error), shapeF func(x, y *shape.Shape) *shape.Shape) interp.FuncBuiltin {
	return func(env engine.Env, call *ir.FuncCallExpr, recv ir.Element, args []ir.Element) ([]ir.Element, error) {
		mat := builtin.Materialiser(env)
		if len(args) != 2 {
			return nil, fmt.Errorf("binary function expects 2 arguments, got %d", len(args))
		}
		x, xShape, err := materialise.Element(mat, args[0])
		if err != nil {
			return nil, err
		}
		y, yShape, err := materialise.Element(mat, args[1])
		if err != nil {
			return nil, err
		}
		node, err := pjrtGraph(env).BinaryFunc(x, y, f)
		if err != nil {
			return nil, err
		}
		outShape := shapeF(xShape, yShape)
		return materialise.ElementFromNode(env.File(), mat, &ops.OutputNode{
			Node:  node,
			Shape: outShape,
		}, call.Type())
	}
}

func pjrtGraph(ctx engine.Env) *pjrtgraph.Graph {
	return ctx.Engine().ArrayOps().Graph().(*pjrtgraph.Graph)
}
