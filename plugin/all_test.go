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

package plugin_test

import (
	"testing"

	"github.com/gx-org/xlapjrt/plugin"
	"github.com/gx-org/gx/build/builder/testbuild"
	core "github.com/gx-org/gx/build/builder/testdata"
	"github.com/gx-org/gx/internal/testing/testrtm"
	"github.com/gx-org/gx/tests"
)

func TestPJRT(t *testing.T) {
	// control.While requires function literals to implement grapheval.SubGrapher,
	// which is not yet supported at runtime.
	core.Sources.Filter = func(name string) bool {
		return name != "control/control.gx"
	}
	bck, err := plugin.NewWithBuilder("cpu", tests.NewBuilder())
	if err != nil {
		t.Fatal(err)
	}
	testbuild.RunFactory(t,
		bck.Builder().Loader().Importers(),
		testrtm.Factory(
			bck,
			tests.All...,
		),
	)
}
