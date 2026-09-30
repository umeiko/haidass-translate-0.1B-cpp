// platform.h — OS abstraction: read-only file mapping + embedded model blob.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace haidass {

// Read-only memory mapping of a file. Non-copyable; unmaps on destruction.
class MappedFile {
public:
    MappedFile() = default;
    ~MappedFile();
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    // Returns false on failure (error message in `err`).
    bool open(const std::string& path, std::string* err);
    void close();

    const uint8_t* data() const { return data_; }
    size_t size() const { return size_; }
    bool is_open() const { return data_ != nullptr; }

private:
    const uint8_t* data_ = nullptr;
    size_t size_ = 0;
#if defined(_WIN32)
    void* file_handle_ = nullptr;    // HANDLE
    void* mapping_handle_ = nullptr; // HANDLE
#else
    int fd_ = -1;
#endif
};

// Embedded model blob (compiled into the binary when built with
// -DHAIDASS_EMBED_MODEL=...). Returns nullptr/0 in file-mode builds.
const uint8_t* embedded_model_data();
size_t embedded_model_size();

// Command-line arguments as UTF-8 strings (on Windows the CRT mangles
// non-ASCII argv into the ANSI codepage, so we rebuild from GetCommandLineW).
// Returns empty vector on POSIX (use argv directly).
std::vector<std::string> platform_utf8_args();

} // namespace haidass
