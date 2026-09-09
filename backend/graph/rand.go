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
	"github.com/gomlx/gopjrt/xlabuilder"
	"github.com/gx-org/backend"
	"github.com/gx-org/backend/shape"
	pjrtgx "github.com/gx-org/xlapjrt"
)

// Random returns the builder for the rand package.
func (g *Graph) Random() backend.RandomBuilder {
	return g
}

// RngBitGenerator takes RNG state and generates the given shape filled with random values, and
// returns the new state plus generated values.
func (g *Graph) RngBitGenerator(state backend.Node, shape *shape.Shape) (backend.Node, backend.Node, error) {
	newState, values, err := xlabuilder.RngBitGenerator(g.xlaHandle(state), pjrtgx.ToShape(shape))
	if err != nil {
		return nil, nil, err
	}
	return g.newNode(newState), g.newNode(values), nil
}
