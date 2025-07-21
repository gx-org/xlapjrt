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
	"github.com/gx-org/backend/ops"
	"github.com/gx-org/backend/platform"
	"github.com/gx-org/gx/build/builder"
	pjrtgraph "github.com/gx-org/xlapjrt/backend/graph"
	pjrtplatform "github.com/gx-org/xlapjrt/backend/platform"
)

type pBackend struct {
	plat *pjrtplatform.Platform
	bld  *builder.Builder
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
func (b *pBackend) Platform() platform.Platform {
	return b.plat
}

// Builder returns the builder to build GX code for the device.
func (b *pBackend) Builder() *builder.Builder {
	return b.bld
}

// NewGraph returns a new XLA computation graph.
func (b *pBackend) NewOps(funcName string) ops.Graph {
	return pjrtgraph.New(b.plat, funcName, nil)
}

// Client returns the PJRT client of the backend.
func (b *pBackend) Client() *pjrt.Client {
	return b.plat.Client()
}
