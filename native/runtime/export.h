#pragma once

// The dylib is built with hidden visibility; this marks what it exports.
#define MR_API __attribute__((visibility("default")))
