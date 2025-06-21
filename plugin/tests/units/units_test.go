// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//	http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package units_test

import (
	"testing"

	"github.com/gx-org/xlapjrt/plugin"
	"github.com/gx-org/gx/api"
	gxtesting "github.com/gx-org/gx/tests/testing"
	"github.com/gx-org/gx/tests"
	"github.com/gx-org/xlapjrt/stdlib"
)

func newRuntime() (*api.Runtime, error) {
	bld := tests.StdlibBuilder(stdlib.Stdlib)
	return plugin.NewWithBuilder("cpu", bld)
}

func TestPJRTUnits(t *testing.T) {
	session := gxtesting.NewUnitSession(newRuntime, tests.FS)
	for _, path := range tests.Units {
		session.TestFolder(t, path)
	}
}
