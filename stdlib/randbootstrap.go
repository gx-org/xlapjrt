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
	"github.com/gx-org/backend/shape"
	"github.com/gx-org/gx/api/values"
	"github.com/gx-org/gx/build/fmterr"
	"github.com/gx-org/gx/build/ir"
	"github.com/gx-org/gx/golang/backend/kernels"
	"github.com/gx-org/gx/golang/binder/gobindings/types"
	"github.com/gx-org/gx/interp/elements"
	"github.com/gx-org/gx/interp"
	"github.com/gx-org/gx/interp/proxies"
	"github.com/gx-org/gx/interp/state"
)

type randBootstrap struct {
	context interp.Context
	call    elements.CallAt
	errF    fmterr.FileSet

	seed elements.NumericalElement
	rand *rand.Rand
	next func() (elements.NumericalElement, error)
}

var _ state.Element = (*randBootstrap)(nil)

func (rb *randBootstrap) Type() ir.Type {
	return ir.BuiltinType{Impl: rb}
}

func (rb *randBootstrap) Flatten() ([]state.Element, error) {
	return []state.Element{rb}, nil
}

func (rb *randBootstrap) Unflatten(handles *elements.Unflattener) (values.Value, error) {
	return nil, fmterr.Internal(errors.Errorf("%T does not support converting device handles into GX values", rb), "")
}

func (rb *randBootstrap) initRand(seed *values.HostArray) error {
	seedValue := types.AtomFromHost[int64](seed)
	rb.rand = rand.New(rand.NewSource(seedValue))
	return nil
}

var uint64Type = ir.TypeFromKind(ir.Uint64Kind)

func (rb *randBootstrap) nextConstant() (elements.NumericalElement, error) {
	next := rb.rand.Uint64()
	expr := &ir.AtomicValueT[uint64]{
		Src: rb.call.Node().Expr(),
		Val: next,
		Typ: uint64Type,
	}
	value := values.AtomIntegerValue(expr.Typ, next)
	return rb.context.Evaluator().ElementFromValue(elements.NewNodeAt[ir.Node](rb.context.File(), expr), value)
}

func (rb *randBootstrap) State() *state.State {
	return rb.context.State()
}

type randBootstrapArg struct {
	seed   elements.ElementWithArrayFromContext
	rb     *randBootstrap
	pValue *proxies.Array
}

func newRandBootstrapArg(ctx interp.Context, rb *randBootstrap, seed elements.ElementWithArrayFromContext) *randBootstrapArg {
	typ := ir.TypeFromKind(ir.Uint64Kind)
	shape := &shape.Shape{DType: dtype.Uint64}
	argFactory := &randBootstrapArg{
		rb:     rb,
		seed:   seed,
		pValue: proxies.NewArray(typ, shape),
	}
	ctx.State().RegisterInit(argFactory)
	return argFactory
}

func (arg *randBootstrapArg) next() (elements.NumericalElement, error) {
	return state.NewArrayArgument(arg, arg.rb.call.ToExprAt(), arg.pValue)
}

func (arg *randBootstrapArg) Init(ctx *elements.CallInputs) error {
	value, err := arg.seed.ArrayFromContext(ctx)
	if err != nil {
		return nil
	}
	hostValue, err := value.ToHost(kernels.Allocator())
	if err != nil {
		return err
	}
	array, ok := hostValue.(*values.HostArray)
	if !ok {
		return errors.Errorf("cannot convert GX argument %T to %T: not supported", value, array)
	}
	return arg.rb.initRand(array)
}

func (arg randBootstrapArg) State() *state.State {
	return arg.rb.State()
}

func (arg randBootstrapArg) Name() string {
	return "randBootstrapArg.next()"
}

func (arg randBootstrapArg) ValueProxy() proxies.Value {
	return arg.pValue
}

func (arg randBootstrapArg) ValueFromContext(ctx *elements.CallInputs) (values.Value, error) {
	val := arg.rb.rand.Uint64()
	return values.AtomIntegerValue[uint64](arg.ValueProxy().Type(), val), nil
}

func evalNewBootstrapGenerator(ctx interp.Context, call elements.CallAt, fn *elements.Func, irFunc *ir.FuncBuiltin, args []state.Element) (output state.Element, err error) {
	bootstrap := &randBootstrap{
		context: ctx,
		call:    call,
		errF:    ctx.FileSet(),
	}
	switch seedNode := args[0].(type) {
	case elements.ElementWithConstant:
		bootstrap.next = bootstrap.nextConstant
		err = bootstrap.initRand(seedNode.NumericalConstant())
	case elements.ElementWithArrayFromContext:
		argFactory := newRandBootstrapArg(ctx, bootstrap, seedNode)
		bootstrap.next = argFactory.next
	default:
		err = errors.Errorf("cannot process seed node: %T not supported", seedNode)
	}
	if err != nil {
		return nil, err
	}
	return elements.NewMethods(bootstrap), nil
}

func evalBootstrapGeneratorNext(ctx interp.Context, call elements.CallAt, fn *elements.Func, irFunc *ir.FuncBuiltin, args []state.Element) (output state.Element, err error) {
	bootStrap := fn.Recv().Element.(*randBootstrap)
	return bootStrap.next()
}
