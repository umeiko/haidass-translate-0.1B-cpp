// platform.cpp — read-only file mapping for Windows and POSIX.
#include "platform.h"

#if defined(_WIN32)
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#    include <shellapi.h>
#else
#    include <fcntl.h>
#    include <sys/mman.h>
#    include <sys/stat.h>
#    include <unistd.h>
#endif

namespace haidass {

MappedFile::~MappedFile() { close(); }

#if defined(_WIN32)

bool MappedFile::open(const std::string& path, std::string* err) {
    close();
    // Convert to wide chars so non-ASCII paths work on Windows.
    int wlen = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
    std::wstring wpath(wlen > 0 ? wlen - 1 : 0, L'\0');
    if (wlen > 0) MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wpath.data(), wlen);

    HANDLE hf = CreateFileW(wpath.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (hf == INVALID_HANDLE_VALUE) {
        if (err) *err = "cannot open file: " + path;
        return false;
    }
    LARGE_INTEGER sz;
    if (!GetFileSizeEx(hf, &sz)) {
        CloseHandle(hf);
        if (err) *err = "cannot get file size: " + path;
        return false;
    }
    HANDLE hm = CreateFileMappingW(hf, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!hm) {
        CloseHandle(hf);
        if (err) *err = "cannot create file mapping: " + path;
        return false;
    }
    const void* p = MapViewOfFile(hm, FILE_MAP_READ, 0, 0, 0);
    if (!p) {
        CloseHandle(hm);
        CloseHandle(hf);
        if (err) *err = "cannot map file: " + path;
        return false;
    }
    file_handle_ = hf;
    mapping_handle_ = hm;
    data_ = static_cast<const uint8_t*>(p);
    size_ = static_cast<size_t>(sz.QuadPart);
    return true;
}

void MappedFile::close() {
    if (data_) UnmapViewOfFile(data_);
    if (mapping_handle_) CloseHandle(static_cast<HANDLE>(mapping_handle_));
    if (file_handle_) CloseHandle(static_cast<HANDLE>(file_handle_));
    data_ = nullptr;
    size_ = 0;
    mapping_handle_ = nullptr;
    file_handle_ = nullptr;
}

std::vector<std::string> platform_utf8_args() {
    std::vector<std::string> out;
    int argc = 0;
    LPWSTR* wargv = CommandLineToArgvW(GetCommandLineW(), &argc);
    if (!wargv) return out;
    for (int i = 0; i < argc; ++i) {
        int len = WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, nullptr, 0, nullptr, nullptr);
        std::string s(len > 0 ? len - 1 : 0, '\0');
        if (len > 0) WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, s.data(), len, nullptr, nullptr);
        out.push_back(std::move(s));
    }
    LocalFree(wargv);
    return out;
}

#else

bool MappedFile::open(const std::string& path, std::string* err) {
    close();
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        if (err) *err = "cannot open file: " + path;
        return false;
    }
    struct stat st;
    if (fstat(fd, &st) != 0) {
        ::close(fd);
        if (err) *err = "cannot stat file: " + path;
        return false;
    }
    if (st.st_size == 0) {
        ::close(fd);
        if (err) *err = "empty file: " + path;
        return false;
    }
    void* p = mmap(nullptr, static_cast<size_t>(st.st_size), PROT_READ, MAP_PRIVATE, fd, 0);
    if (p == MAP_FAILED) {
        ::close(fd);
        if (err) *err = "cannot mmap file: " + path;
        return false;
    }
    fd_ = fd;
    data_ = static_cast<const uint8_t*>(p);
    size_ = static_cast<size_t>(st.st_size);
    return true;
}

void MappedFile::close() {
    if (data_) munmap(const_cast<uint8_t*>(data_), size_);
    if (fd_ >= 0) ::close(fd_);
    data_ = nullptr;
    size_ = 0;
    fd_ = -1;
}

std::vector<std::string> platform_utf8_args() { return {}; }

#endif

} // namespace haidass
