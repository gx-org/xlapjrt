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

// Package plugin creates a PJRT GX backend from a PJRT plugin name.
package plugin

import (
	"fmt"

	"github.com/gomlx/gopjrt/pjrt"
	"github.com/gx-org/gx/api"
	"github.com/gx-org/gx/build/builder"
	"github.com/gx-org/gx/build/importers/embedpkg"
	"github.com/gx-org/gx/build/importers"
	"github.com/gx-org/gx/build/importers/localfs"
	"github.com/gx-org/gx/stdlib"
	"github.com/gx-org/xlapjrt/backend"
	pjrtstdlib "github.com/gx-org/xlapjrt/stdlib"
)

// New returns a new PJRT runtime given a plugin name.
func New(name string) (*api.Runtime, error) {
	localImporter, err := localfs.New("")
	if err != nil {
		return nil, err
	}
	var importer importers.Importer
	if localImporter != nil {
		importer = localImporter
	} else {
		// No Go module can be found from the current working directory,
		// fallback to embedded files in the binary.
		importer = embedpkg.New()
	}
	bld := builder.New(importers.NewCacheLoader(
		stdlib.Importer(pjrtstdlib.Stdlib),
		importer,
	))
	return NewWithBuilder(name, bld)
}

// NewWithBuilder creates PJRT GX runtime given a GX builder, a plugin name, and client options.
func NewWithBuilder(name string, bld *builder.Builder) (*api.Runtime, error) {
	plugin, err := pjrt.GetPlugin(name)
	if err != nil {
		return nil, fmt.Errorf("cannot load PJRT plugin %q: %v", name, err)
	}
	pjrtBackend, err := backend.New(bld, plugin)
	if err != nil {
		return nil, err
	}
	return api.NewRuntime(pjrtBackend, bld), nil
}
