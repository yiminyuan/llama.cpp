#include "common.h"

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#undef NDEBUG
#include <cassert>

static std::string groups_to_str(const common_tp_groups & groups) {
    if (groups.empty()) {
        return "<all>";
    }
    std::string res;
    for (size_t i = 0; i < groups.size(); i++) {
        if (i > 0) {
            res += "/";
        }
        for (size_t j = 0; j < groups[i].size(); j++) {
            if (j > 0) {
                res += "+";
            }
            res += groups[i][j];
        }
    }
    return res;
}

static void assert_parse(const std::string & value, const common_tp_groups & expected) {
    const common_tp_groups actual = common_tp_groups_parse(value);
    if (actual != expected) {
        fprintf(stderr, "test-tp-groups: '%s' parsed to '%s', expected '%s'\n",
                value.c_str(), groups_to_str(actual).c_str(), groups_to_str(expected).c_str());
        assert(false);
    }
}

static void assert_parse_fails(const std::string & value) {
    common_tp_groups actual;
    try {
        actual = common_tp_groups_parse(value);
    } catch (const std::invalid_argument &) {
        return;
    }
    fprintf(stderr, "test-tp-groups: '%s' parsed to '%s', expected it to be rejected\n",
            value.c_str(), groups_to_str(actual).c_str());
    assert(false);
}

static void test_parse_valid(void) {
    printf("test-tp-groups: parse valid specs\n");

    assert_parse("all", {});

    assert_parse("ROCm0",                   {{"ROCm0"}});
    assert_parse("ROCm0+ROCm1",             {{"ROCm0", "ROCm1"}});
    assert_parse("ROCm0+ROCm1+ROCm2+ROCm3", {{"ROCm0", "ROCm1", "ROCm2", "ROCm3"}});

    assert_parse("ROCm0/RPC0",                   {{"ROCm0"}, {"RPC0"}});
    assert_parse("ROCm0+ROCm1+ROCm2+ROCm3/RPC0", {{"ROCm0", "ROCm1", "ROCm2", "ROCm3"}, {"RPC0"}});
    assert_parse("ROCm0+ROCm1/ROCm2+ROCm3",      {{"ROCm0", "ROCm1"}, {"ROCm2", "ROCm3"}});
    assert_parse("CUDA0/CUDA1/CUDA2",            {{"CUDA0"}, {"CUDA1"}, {"CUDA2"}});

    // order is kept as written, so errors can echo it back
    assert_parse("ROCm3+ROCm0", {{"ROCm3", "ROCm0"}});

    assert_parse(" ROCm0 + ROCm1 / RPC0 ", {{"ROCm0", "ROCm1"}, {"RPC0"}});
    assert_parse("\tROCm0\t+\tROCm1\t",    {{"ROCm0", "ROCm1"}});

    // "all" is a keyword only as the whole value, elsewhere it is a device name
    assert_parse("all/ROCm0", {{"all"}, {"ROCm0"}});
}

static void test_parse_malformed(void) {
    printf("test-tp-groups: reject malformed specs\n");

    assert_parse_fails("");
    assert_parse_fails("   ");

    // separators with nothing around them
    assert_parse_fails("/");
    assert_parse_fails("+");
    assert_parse_fails("+/");

    // empty group
    assert_parse_fails("/ROCm0");
    assert_parse_fails("ROCm0/");
    assert_parse_fails("ROCm0//RPC0");

    // empty device name
    assert_parse_fails("+ROCm0");
    assert_parse_fails("ROCm0+");
    assert_parse_fails("ROCm0++ROCm1");
    assert_parse_fails("ROCm0+/RPC0");

    // whitespace-only components are empty too
    assert_parse_fails("ROCm0+ /RPC0");
    assert_parse_fails("ROCm0/ /RPC0");
}

static void test_parse_duplicates(void) {
    printf("test-tp-groups: reject repeated devices\n");

    // a device cannot appear twice in one group
    assert_parse_fails("ROCm0+ROCm0");
    assert_parse_fails("ROCm0+ROCm1+ROCm0");

    // nor be shared between groups, which would put it in two tensor-parallel groups at once
    assert_parse_fails("ROCm0/ROCm0");
    assert_parse_fails("ROCm0+ROCm1/ROCm1+ROCm2");
    assert_parse_fails("ROCm0+ROCm1/RPC0/ROCm0");

    // the check is on the trimmed name, so spacing cannot hide a duplicate
    assert_parse_fails("ROCm0 + ROCm0");
    assert_parse_fails("ROCm0 / ROCm0 ");
}

int main(void) {
    try {
        test_parse_valid();
        test_parse_malformed();
        test_parse_duplicates();
    } catch (const std::exception & e) {
        fprintf(stderr, "test-tp-groups: exception: %s\n", e.what());
        return 1;
    }

    printf("test-tp-groups: all tests OK\n\n");

    return 0;
}
