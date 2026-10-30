#pragma once

// NSSharedPtr.hpp uses std::is_convertible_v without including <type_traits>.
// clang-format off
#include <type_traits>
#include <Foundation/NSSharedPtr.hpp>
// clang-format on

// Members of this type need the pointee complete wherever they are destroyed,
// so classes holding one define their destructor out of line.
