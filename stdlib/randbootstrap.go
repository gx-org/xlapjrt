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
	"math/rand"

	"github.com/pkg/errors"
	"github.com/gx-org/backend/dtype"
	"github.com/gx-org/backend/platform"
	"github.com/gx-org/backend/shape"
	"github.com/gx-org/gx/api/values"
	"github.com/gx-org/gx/build/fmterr"
	"github.com/gx-org/gx/build/ir"
	"github.com/gx-org/gx/golang/backend/kernels"
	"github.com/gx-org/gx/golang/binder/gobindings/types"
	"github.com/gx-org/gx/interp"
	"github.com/gx-org/gx/interp/state"
)

type randBootstrap struct {
	context interp.Context
	call    state.CallAt
	errF    fmterr.FileSet

	seed state.NumericalElement
	rand *rand.Rand
	next func() (state.NumericalElement, error)
}

var _ state.Element = (*randBootstrap)(nil)

func (rb *randBootstrap) Type() ir.Type {
	return ir.BuiltinType{Impl: rb}
}

func (rb *randBootstrap) initRand(seed *values.HostArray) error {
	seedValue := types.AtomFromHost[int64](seed)
	rb.rand = rand.New(rand.NewSource(seedValue))
	return nil
}

var uint64Type = ir.ToAtomic(ir.Uint64Kind)

func (rb *randBootstrap) nextConstant() (state.NumericalElement, error) {
	next := rb.rand.Uint64()
	return state.NewAtomicLiteral[uint64](rb.context.State(), uint64Type, next)
}

func (rb *randBootstrap) State() *state.State {
	return rb.context.State()
}

type randBootstrapArg struct {
	seedArg *state.ArgGX
	rb      *randBootstrap
}

func (arg *randBootstrapArg) next() (state.NumericalElement, error) {
	ctx := arg.rb.context
	return ctx.State().NewArgElement("randBootstrap.next", arg.seedArg.Field().ToExprAt(), arg)
}

func (arg randBootstrapArg) Shape() *shape.Shape {
	return &shape.Shape{DType: dtype.Uint64}
}

func (arg randBootstrapArg) Type() ir.Type {
	return ir.ScalarTypeK(ir.Uint64Kind)
}

func (arg randBootstrapArg) Init(ctx state.Context) error {
	value := arg.seedArg.Value(ctx)
	array, ok := value.(*values.HostArray)
	if !ok {
		return errors.Errorf("cannot convert GX argument %T to %T: not supported", value, array)
	}
	return arg.rb.initRand(array)
}

func (arg randBootstrapArg) ToDeviceHandle(device platform.Device, ctx state.Context) (platform.DeviceHandle, error) {
	val := arg.rb.rand.Uint64()
	handle := kernels.ToAlgebraicAtom(val)
	return device.Send(handle.Buffer(), handle.Shape())
}

func evalNewBootstrapGenerator(ctx interp.Context, call state.CallAt, fn *state.Func, irFunc *ir.FuncBuiltin, args []state.Element) (output state.Element, err error) {
	bootstrap := &randBootstrap{
		context: ctx,
		call:    call,
		errF:    ctx.FileSet(),
	}
	switch seedNode := args[0].(type) {
	case state.ElementWithConstant:
		bootstrap.next = bootstrap.nextConstant
		err = bootstrap.initRand(seedNode.NumericalConstant())
	case *state.ArgGX:
		argFactory := randBootstrapArg{rb: bootstrap, seedArg: seedNode}
		ctx.State().RegisterInit(argFactory)
		bootstrap.next = argFactory.next
	default:
		err = errors.Errorf("cannot process seed node: %T not supported", seedNode)
	}
	if err != nil {
		return nil, err
	}
	methods := ctx.State().Methods(bootstrap)
	return methods, nil
}

func evalBootstrapGeneratorNext(ctx interp.Context, call state.CallAt, fn *state.Func, irFunc *ir.FuncBuiltin, args []state.Element) (output state.Element, err error) {
	bootStrap := fn.Recv().Element.(*randBootstrap)
	return bootStrap.next()
}
