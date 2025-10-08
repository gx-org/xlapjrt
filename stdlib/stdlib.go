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
	"github.com/gx-org/gx/interp/elements"
	"github.com/gx-org/gx/interp/evaluator"
	"github.com/gx-org/gx/interp/fun"
	"github.com/gx-org/gx/interp"
	"github.com/gx-org/gx/interp/materialise"
	"github.com/gx-org/gx/stdlib/builtin"
	"github.com/gx-org/gx/stdlib/impl"
	pjrtgraph "github.com/gx-org/xlapjrt/backend/graph"
)

// Stdlib is the PJRT implementation of the standard library.
var Stdlib = &impl.Stdlib{
	Dtype: impl.Dtype{
		Reinterpret: evalReinterpret,
	},
	Math: impl.Math{
		Abs:      xlaUnaryFunc(xlabuilder.Abs),
		Ceil:     xlaUnaryFunc(xlabuilder.Ceil),
		Erf:      xlaUnaryFunc(xlabuilder.Erf),
		Expm1:    xlaUnaryFunc(xlabuilder.Expm1),
		Floor:    xlaUnaryFunc(xlabuilder.Floor),
		Log1p:    xlaUnaryFunc(xlabuilder.Log1p),
		Logistic: xlaUnaryFunc(xlabuilder.Logistic),
		Max:      xlaBinaryFunc(xlabuilder.Max, minmaxDType),
		Min:      xlaBinaryFunc(xlabuilder.Min, minmaxDType),
		Pow:      xlaBinaryFunc(xlabuilder.Pow, firstArgument),
		Round:    xlaUnaryFunc(xlabuilder.Round),
		Rsqrt:    xlaUnaryFunc(xlabuilder.Rsqrt),
		Sign:     xlaUnaryFunc(xlabuilder.Sign),
		Sqrt:     xlaUnaryFunc(xlabuilder.Sqrt),
	},
	Num: impl.Num{
		Iota:      evalIota,
		Transpose: evalTranspose,
		Einsum:    evalEinsum,
		MatMul:    xlaBinaryFunc(xlabuilder.Dot, matmulShape),
		Sum:       xlaReductionFunc(xlabuilder.ReduceSum),
		ReduceMax: xlaReductionFunc(xlabuilder.ReduceMax),
		Argmax:    evalArgmax,
	},
	Rand: impl.Rand{
		PhiloxUint32: evalPhiloxUint32,
		PhiloxUint64: evalPhiloxUint64,
	},
	Shapes: impl.Shapes{
		Concat: evalConcat,
		Len:    evalLen,
		Split:  evalSplit,
		Gather: evalGather,
	},
}

func xlaUnaryFunc(f func(*xlabuilder.Op) (*xlabuilder.Op, error)) interp.FuncBuiltin {
	return func(env evaluator.Env, call elements.CallAt, fn fun.Func, irFunc *ir.FuncBuiltin, args []ir.Element) ([]ir.Element, error) {
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
		return mat.ElementsFromNodes(call.File(), call.Node(), &ops.OutputNode{
			Node:  node,
			Shape: xShape,
		})
	}
}

func firstArgument(x, _ *shape.Shape) *shape.Shape {
	return x
}

func minmaxDType(x, y *shape.Shape) *shape.Shape {
	target := &shape.Shape{
		DType:       x.DType,
		AxisLengths: x.AxisLengths,
	}
	if y.AxisLengths != nil {
		target.AxisLengths = y.AxisLengths
	}
	return target
}

func matmulShape(x, y *shape.Shape) *shape.Shape {
	var lengths []int
	if len(x.AxisLengths) > 0 {
		lengths = append(lengths, x.AxisLengths[:len(x.AxisLengths)-1]...)
	}
	if len(y.AxisLengths) > 0 {
		lengths = append(lengths, y.AxisLengths[1:]...)
	}
	return &shape.Shape{
		DType:       x.DType,
		AxisLengths: lengths,
	}
}

func xlaBinaryFunc(f func(x *xlabuilder.Op, y *xlabuilder.Op) (*xlabuilder.Op, error), shapeF func(x, y *shape.Shape) *shape.Shape) interp.FuncBuiltin {
	return func(env evaluator.Env, call elements.CallAt, fn fun.Func, irFunc *ir.FuncBuiltin, args []ir.Element) ([]ir.Element, error) {
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
		return mat.ElementsFromNodes(call.File(), call.Node(), &ops.OutputNode{
			Node:  node,
			Shape: outShape,
		})
	}
}

func pjrtGraph(ctx evaluator.Env) *pjrtgraph.Graph {
	return ctx.Evaluator().ArrayOps().Graph().(*pjrtgraph.Graph)
}
