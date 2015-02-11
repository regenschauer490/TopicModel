ifeq "$(strip $(OBJ_DIR))" ""
  OBJ_DIR = .
endif

MAIN_DIR = ../../SigTM

TEST_DIR = $(MAIN_DIR)/example
TEST_FILES = $(wildcard $(TEST_DIR)/*.cpp) 
TEST_FNAMES = $(subst $(TEST_DIR)/,,$(TEST_FILES))
TEST_TMP = $(TEST_FNAMES:.cpp=.o)
TEST_OBJS = $(addprefix $(OBJ_DIR)/,$(TEST_TMP))

MODEL_DIR = $(MAIN_DIR)/lib/model
MODEL_FILES = $(wildcard $(MODEL_DIR)/*.cpp)
MODEL_FNAMES = $(subst $(MODEL_DIR)/,,$(MODEL_FILES))
MODEL_TMP = $(MODEL_FNAMES:.cpp=.o)
MODEL_OBJS = $(addprefix $(OBJ_DIR)/,$(MODEL_TMP))

MAIN_OBJ = $(addprefix $(OBJ_DIR)/,main.o)
OBJS = $(TEST_OBJS) $(MODEL_OBJS) $(MAIN_OBJ)
DEPENDS  = $(OBJS:.o=.d)


$(TARGET) : $(OBJS) $(LIBS)
	@[ -d $(BIN_DIR) ] || mkdir -p $(BIN_DIR)
	$(COMPILER) -o $(TARGET0) $^ $(LDFLAGS)

$(OBJ_DIR)/main.o : $(MAIN_DIR)/main.cpp
	@[ -d $(OBJ_DIR) ] || mkdir -p $(OBJ_DIR)
	$(COMPILER) $(CFLAGS) $(INCLUDE) -c $< -o $@

$(OBJ_DIR)/%.o : $(TEST_DIR)/%.cpp
	@[ -d $(OBJ_DIR) ] || mkdir -p $(OBJ_DIR)
	$(COMPILER) $(CFLAGS) $(INCLUDE) -c $< -o $@
	
$(OBJ_DIR)/%.o : $(MODEL_DIR)/%.cpp
	@[ -d $(OBJ_DIR) ] || mkdir -p $(OBJ_DIR)
	$(COMPILER) $(CFLAGS) $(INCLUDE) -c $< -o $@

$(OBJ_DIR)/main.o : $(TEST_DIR)/example.h

$(OBJ_DIR)/%.o : $(TEST_DIR)/example.h

$(OBJ_DIR)/%.o : $(addprefix $(MODEL_DIR)/,%.h)
 
all: clean $(TARGET)
 
clean:
	rm -f $(OBJS) $(DEPENDS) $(TARGET)
	@rmdir --ignore-fail-on-non-empty `readlink -f $(OBJ_DIR)`
 
-include $(DEPENDS)
