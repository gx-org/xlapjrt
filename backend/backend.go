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

// Package backend provides a XLA backend to GX given a gomlx XLA client.
package backend

import (
	"google3/third_party/golang/github_com/gomlx/compute/v/v0/compute"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/gopjrt/pjrt"
	"github.com/gx-org/backend"
	"github.com/gx-org/gx/build/builder"
	pjrtgraph "github.com/gx-org/xlapjrt/backend/graph"
	pjrtplatform "github.com/gx-org/xlapjrt/backend/platform"
)

type (
	pBackend struct {
		*pjrtplatform.Platform
		bld *builder.Builder
	}

	builderImpl struct {
		name    string
		main    *pjrtgraph.Graph
		devices []backend.DeviceNum
	}
)

var (
	_ backend.Backend = (*pBackend)(nil)
	_ backend.Builder = (*builderImpl)(nil)
)

func (b *builderImpl) Name() string {
	return b.name
}

func (b *builderImpl) Main() backend.Function {
	return b.main
}

func (b *builderImpl) NewFunction(name string) (backend.Function, error) {
	return b.main.NewFunction(name)
}

func (b *builderImpl) OpShape(op compute.Value) (shapes.Shape, error) {
	return b.main.Shape(op)
}

func (b *builderImpl) DeviceAssignment(devices ...backend.DeviceNum) error {
	b.devices = devices
	return nil
}

func (b *builderImpl) Compile() (backend.Executable, error) {
	var dev backend.DeviceNum
	if len(b.devices) > 0 {
		dev = b.devices[0]
	}
	return b.main.Compile(dev)
}

// New returns a new PJRT backend.
func New(builder *builder.Builder, plugin *pjrt.Plugin) (backend.Backend, error) {
	client, err := plugin.NewClient(nil)
	if err != nil {
		return nil, err
	}
	bck := &pBackend{
		bld: builder,
	}
	bck.Platform = pjrtplatform.New(client, bck)
	return bck, nil
}

// Builder returns a new XLA computation builder.
func (b *pBackend) Builder(funcName string) backend.Builder {
	bld := &builderImpl{
		name: funcName,
	}
	bld.main = pjrtgraph.New(b.Platform, bld, funcName)
	return bld
}
