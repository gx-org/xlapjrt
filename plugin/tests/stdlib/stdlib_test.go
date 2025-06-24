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
package stdlib_test

import (
	"testing"

	"github.com/gx-org/xlapjrt/plugin"
	gxtesting "github.com/gx-org/gx/tests/testing"
	"github.com/gx-org/gx/tests"
	"github.com/gx-org/xlapjrt/stdlib"
)

func TestPJRTStdlib(t *testing.T) {
	t.SkipNow() // TODO(degris): FIX ASAP
	bld := tests.StdlibBuilder(stdlib.Stdlib)
	bck, err := plugin.NewWithBuilder("cpu", bld)
	if err != nil {
		t.Fatal(err)
	}
	session := gxtesting.NewSession(bck, tests.FS)
	for _, path := range tests.All {
		session.TestFolder(t, path)
	}
}
