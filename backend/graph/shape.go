// Copyright 2026 Google LLC
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

package graph

import (
	"google3/third_party/golang/github_com/gomlx/compute/v/v0/compute"
	"github.com/gomlx/gopjrt/xlabuilder"
)

// Concatenate concatenates multiple arrays into a single array.
func (g *Graph) Concatenate(axis int, operands ...compute.Value) (compute.Value, error) {
	inputs, err := g.xlaHandles(operands)
	if err != nil {
		return nil, err
	}

	xlaOp, err := xlabuilder.Concatenate(axis, inputs...)
	if err != nil {
		return nil, err
	}
	return g.newNode(xlaOp), nil
}
