// embed_none.cpp — file-mode build: no embedded model.
#include "platform.h"

namespace haidass {

const uint8_t* embedded_model_data() { return nullptr; }
size_t embedded_model_size() { return 0; }

} // namespace haidass
