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
	"github.com/gx-org/backend/platform"
)

// Platform is the PJRT platform.
type Platform struct {
	clt    *pjrt.Client
	device *Device
}

// New PJRT platform.
func New(clt *pjrt.Client) *Platform {
	plat := &Platform{clt: clt}
	plat.device = &Device{plat: plat}
	return plat
}

// Name of the platform.
func (plat *Platform) Name() string {
	return "pjrt"
}

// Device returns a device given its ID.
// The same pointer will be returned for the same ID.
// Consequently, it is valid to compare pointers to check that two devices are the same.
func (plat *Platform) Device(ordinal int) (platform.Device, error) {
	return plat.device, nil
}

// Client returns the PJRT client.
func (plat *Platform) Client() *pjrt.Client {
	return plat.clt
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
