// Copyright 2025 Google LLC
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

package xlapjrt

//go:generate go tool ccgx link
//go:generate go tool packager --gx_package=github.com/gx-org/xlapjrt/gx
//go:generate go tool cgo -exportheader cgx/cgx.cgo.h cgx/cgx.go
//go:generate go tool cgo -exportheader testing/testing.cgo.h testing/testing.go
