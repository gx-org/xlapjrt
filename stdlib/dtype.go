// Copyright 2025 Google LLC
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
	"fmt"

	"github.com/gx-org/backend/ops"
	"github.com/gx-org/backend/shape"
	"github.com/gx-org/gx/build/ir"
	"github.com/gx-org/gx/interp/elements"
	"github.com/gx-org/gx/interp/evaluator"
	"github.com/gx-org/gx/interp"
	"github.com/gx-org/gx/interp/materialise"
)

func evalReinterpret(ctx evaluator.Context, call elements.CallAt, fn interp.Func, irFunc *ir.FuncBuiltin, args []ir.Element) ([]ir.Element, error) {
	argNode, _, err := materialise.Element(ctx.Materialiser(), args[0])
	if err != nil {
		return nil, err
	}
	retType := call.Node().Callee.T.Results.List[0].Type.Typ
	arrayType, ok := ir.Underlying(retType).(ir.ArrayType)
	if !ok {
		return nil, fmt.Errorf("%T is not an array type", retType)
	}
	dtype := arrayType.DataType().Kind().DType()
	op, err := pjrtGraph(ctx).Bitcast(argNode, dtype)
	if err != nil {
		return nil, err
	}
	return ctx.Materialiser().ElementsFromNodes(call.File(), call.Node(), &ops.OutputNode{
		Node: op,
		Shape: &shape.Shape{
			DType:       dtype,
			AxisLengths: op.(interface{ PJRTDims() []int }).PJRTDims(),
		},
	})
}
