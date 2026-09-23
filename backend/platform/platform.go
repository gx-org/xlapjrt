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

// Package platform provides the pjrt platform for GX.
package platform

import (
	"github.com/gomlx/gopjrt/pjrt"
	"github.com/gx-org/backend"
	"github.com/gx-org/backend/shapes"
)

// Platform is the PJRT backend.
type Platform struct {
	clt *pjrt.Client
}

// New PJRT backend.
func New(clt *pjrt.Client) *Platform {
	return &Platform{clt: clt}
}

// Name of the backend.
func (plat *Platform) Name() string {
	return "pjrt"
}

// Send raw data to the device.
func (plat *Platform) Send(dev backend.DeviceNum, data []byte, sh *shapes.Shape) (backend.DeviceHandle, error) {
	return plat.send(dev, data, sh)
}

// Client returns the PJRT client.
func (plat *Platform) Client() *pjrt.Client {
	return plat.clt
}

// Finalize everything linked to the backend.
// It is invalid to use any device from the platform after this call.
func (plat *Platform) Finalize() error {
	return plat.clt.Destroy()
}

func toInt32(input []int) []int32 {
	result := make([]int32, len(input))
	for i, n := range input {
		result[i] = int32(n)
	}
	return result
}

func toInt(input []int32) []int {
	result := make([]int, len(input))
	for i, n := range input {
		result[i] = int(n)
	}
	return result
}
