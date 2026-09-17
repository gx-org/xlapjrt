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
	"github.com/gomlx/gopjrt/pjrt"
	"github.com/gx-org/backend"
	"github.com/gx-org/backend/shape"
	"github.com/gx-org/gx/build/builder"
	pjrtgraph "github.com/gx-org/xlapjrt/backend/graph"
	pjrtplatform "github.com/gx-org/xlapjrt/backend/platform"
)

type (
	pBackend struct {
		plat *pjrtplatform.Platform
		bld  *builder.Builder
	}

	builderImpl struct {
		name string
		main *pjrtgraph.Graph
	}
)

var _ backend.Builder = (*builderImpl)(nil)

func (b *builderImpl) Name() string {
	return b.name
}

func (b *builderImpl) Main() backend.Function {
	return b.main
}

func (b *builderImpl) Compile(dev backend.DeviceNum, output, traced []*backend.OutputNode, params []*shape.Shape) (backend.Executable, error) {
	return b.main.Compile(dev, output, traced, params)
}

// New returns a new PJRT backend.
func New(builder *builder.Builder, plugin *pjrt.Plugin) (backend.Backend, error) {
	client, err := plugin.NewClient(nil)
	if err != nil {
		return nil, err
	}
	return &pBackend{
		bld:  builder,
		plat: pjrtplatform.New(client),
	}, nil
}

// Platform used by the backend.
func (b *pBackend) Platform() backend.Platform {
	return b.plat
}

// Builder returns a new XLA computation builder.
func (b *pBackend) Builder(funcName string) (backend.Builder, error) {
	fn, err := pjrtgraph.New(b.plat, funcName, nil)
	if err != nil {
		return nil, err
	}
	return &builderImpl{
		name: funcName,
		main: fn,
	}, nil
}

// Client returns the PJRT client of the backend.
func (b *pBackend) Client() *pjrt.Client {
	return b.plat.Client()
}

// Release the backend.
func (b *pBackend) Finalize() error {
	err := b.plat.Finalize()
	b.plat = nil
	return err
}
