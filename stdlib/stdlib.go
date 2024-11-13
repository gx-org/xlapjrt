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
	"github.com/gx-org/backend/shape"
	"github.com/gx-org/gx/build/ir"
	"github.com/gx-org/gx/interp"
	"github.com/gx-org/gx/interp/state"
	"github.com/gx-org/gx/stdlib/impl"
	pjrtgraph "github.com/gx-org/xlapjrt/backend/graph"
)

// Stdlib is the PJRT implementation of the standard library.
var Stdlib = &impl.Stdlib{
	Math: impl.Math{
		Pow:  xlaBinaryFunc(xlabuilder.Pow, firstArgument),
		Exp:  xlaUnaryFunc(xlabuilder.Exp),
		Log:  xlaUnaryFunc(xlabuilder.Log),
		Min:  xlaBinaryFunc(xlabuilder.Min, minmaxDType),
		Max:  xlaBinaryFunc(xlabuilder.Max, minmaxDType),
		Cos:  xlaUnaryFunc(xlabuilder.Cos),
		Sin:  xlaUnaryFunc(xlabuilder.Sin),
		Sqrt: xlaUnaryFunc(xlabuilder.Sqrt),
		Tanh: xlaUnaryFunc(xlabuilder.Tanh),
	},
	Num: impl.Num{
		Iota:      evalIota,
		IotaFull:  evalIotaFull,
		Transpose: evalTranspose,
		Einsum:    evalEinsum,
		MatMul:    xlaBinaryFunc(xlabuilder.Dot, matmulShape),
		Sum:       xlaReductionFunc(xlabuilder.ReduceSum),
		ReduceMax: xlaReductionFunc(xlabuilder.ReduceMax),
		Argmax:    evalArgmax,
	},
	Rand: impl.Rand{
		BootstrapGeneratorNew:  evalNewBootstrapGenerator,
		BootstrapGeneratorNext: evalBootstrapGeneratorNext,
		PhiloxUint64:           evalPhiloxUint64,
	},
	Shapes: impl.Shapes{
		Expand: evalExpand,
		Concat: evalConcat,
		Len:    evalLen,
		Split:  evalSplit,
		Gather: evalGather,
	},
}

func xlaUnaryFunc(f func(*xlabuilder.Op) (*xlabuilder.Op, error)) interp.FuncBuiltin {
	return func(ctx interp.Context, call state.CallAt, fn *state.Func, irFunc *ir.FuncBuiltin, args []state.Element) (state.Element, error) {
		if len(args) != 1 {
			return nil, fmt.Errorf("unary function expects 1 argument, got %d", len(args))
		}
		x, xShape, err := state.NodeFromElement(args[0])
		if err != nil {
			return nil, err
		}
		node, err := pjrtGraph(ctx).NewUnaryFunc(x, f)
		if err != nil {
			return nil, err
		}
		return ctx.State().ElementFromNode(call.ToExprAt(), node, xShape)
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
	lengths := []int{}
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
	return func(ctx interp.Context, call state.CallAt, fn *state.Func, irFunc *ir.FuncBuiltin, args []state.Element) (state.Element, error) {
		if len(args) != 2 {
			return nil, fmt.Errorf("binary function expects 2 arguments, got %d", len(args))
		}
		x, xShape, err := state.NodeFromElement(args[0])
		if err != nil {
			return nil, err
		}
		y, yShape, err := state.NodeFromElement(args[1])
		if err != nil {
			return nil, err
		}
		node, err := pjrtGraph(ctx).NewBinaryFunc(x, y, f)
		if err != nil {
			return nil, err
		}
		outShape := shapeF(xShape, yShape)
		return ctx.State().ElementFromNode(call.ToExprAt(), node, outShape)
	}
}

func pjrtGraph(ctx interp.Context) *pjrtgraph.Graph {
	return ctx.State().BackendGraph().(*pjrtgraph.Graph)
}
