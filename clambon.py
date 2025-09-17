import re
import time
import random
import math
import importlib

active_extensions = []

def load_extension(module_name):
    try:
        mod = importlib.import_module(module_name)
        active_extensions.append(mod)
        if hasattr(mod, "on_import"):
            mod.on_import()
        print(f"拡張機能「{module_name}」を読み込みました")
    except ImportError as e:
        print(f"拡張機能「{module_name}」の読み込みに失敗しました: {e}")

def process_extensions(line: str) -> str:
    for ext in active_extensions:
        if hasattr(ext, "process_line"):
            line = ext.process_line(line)
    return line

_STR_RE = re.compile(r'"[^"\\]*(?:\\.[^"\\]*)*"')
_NUMERIC_RE = re.compile(r'^-?\d+(?:\.\d+)?$')
_FUNC_CALL_RE = re.compile(r'^\w+\s*\(.*\)$', re.UNICODE)

def _strip_strings(s: str) -> str:
    return _STR_RE.sub('""', s)

def _ensure_only_supported_comparators(expr: str):
    no_str = _strip_strings(expr)
    if re.search(r'!=|>=|<=', no_str):
        raise ValueError("未サポートの比較演算子を使用しています（使用可: ==, >, <）")

def _replace_word_outside_strings(s: str, word: str, repl: str) -> str:
    out = []
    i = 0
    n = len(s)
    blen = len(word)
    while i < n:
        if s[i] == '"':
            j = i + 1
            while j < n:
                if s[j] == '\\':
                    j += 2
                    continue
                if s[j] == '"':
                    j += 1
                    break
                j += 1
            out.append(s[i:j]); i = j
        else:
            m = re.match(r'\b' + re.escape(word) + r'\b', s[i:])
            if m:
                out.append(repl); i += blen
            else:
                out.append(s[i]); i += 1
    return ''.join(out)

def _is_number(x): return isinstance(x, (int, float)) and not isinstance(x, bool)
def _is_string(x): return isinstance(x, str)
def _is_bool(x):   return isinstance(x, bool)

def _same_clam_type(a, b):
    if _is_number(a) and _is_number(b): return True
    if _is_string(a) and _is_string(b): return True
    if _is_bool(a)   and _is_bool(b):   return True
    return False

def _find_matching_paren(s: str, open_i: int) -> int:
    depth, i, n = 1, open_i + 1, len(s)
    in_str = False
    while i < n:
        c = s[i]
        if in_str:
            if c == '\\':
                i += 2
                continue
            if c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
                if depth == 0:
                    return i
        i += 1
    return -1

def _split_args(arg_s: str):
    args, cur, depth, in_str = [], [], 0, False
    i, n = 0, len(arg_s)
    while i < n:
        c = arg_s[i]
        if in_str:
            cur.append(c)
            if c == '\\' and i + 1 < n:
                cur.append(arg_s[i+1]); i += 1
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True; cur.append(c)
            elif c == '(':
                depth += 1; cur.append(c)
            elif c == ')':
                depth -= 1; cur.append(c)
            elif c == ',' and depth == 0:
                args.append(''.join(cur).strip()); cur = []
            else:
                cur.append(c)
        i += 1
    if cur: args.append(''.join(cur).strip())
    return args

def _replace_func(expr: str, name: str, handler) -> str:
    i, n, out = 0, len(expr), []
    in_str = False
    while i < n:
        if in_str:
            out.append(expr[i])
            if expr[i] == '\\' and i + 1 < n:
                out.append(expr[i+1]); i += 2; continue
            if expr[i] == '"':
                in_str = False
            i += 1; continue
        c = expr[i]
        if c == '"':
            in_str = True; out.append(c); i += 1; continue
        if expr.startswith(name + '(', i):
            if i > 0 and (expr[i-1].isalnum() or expr[i-1] == '_'):
                out.append(c); i += 1; continue
            open_i = i + len(name)
            close_i = _find_matching_paren(expr, open_i)
            if close_i == -1:
                out.append(c); i += 1; continue
            arg_s = expr[open_i+1:close_i]
            out.append(handler(arg_s))
            i = close_i + 1
        else:
            out.append(c); i += 1
    return ''.join(out)

def clambon_bool_replacer(expr, variables):
    return expr.strip()

class ReturnSignal(Exception):
    def __init__(self, value):
        self.value = value

def clambon_eval(expr, variables):
    expr = expr.strip()

    if '__timer_start' in variables:
        elapsed = time.time() - variables['__timer_start']
        variables['timer'] = int(elapsed)

    expr = _replace_word_outside_strings(expr, 'true', 'True')
    expr = _replace_word_outside_strings(expr, 'false', 'False')

    _ensure_only_supported_comparators(expr)

    pattern_contains = r'(\w+)\.contains\(((?:[^()]|\([^()]*\))*)\)'
    def contains_repl(m):
        name = m.group(1)
        value_str = m.group(2).strip()
        if name not in variables:
            raise Exception(f"{name}が見つかりません")
        base = variables[name]
        try:
            v = clambon_eval(value_str, variables)
        except Exception:
            v = value_str
        if isinstance(base, list):
            return "True" if v in base else "False"
        elif isinstance(base, str):
            if not isinstance(v, str):
                v = str(v)
            return "True" if v in base else "False"
        else:
            raise Exception(f"{name}は配列でも文字列でもありません")
    prev_expr = None
    while prev_expr != expr:
        prev_expr = expr
        expr = re.sub(pattern_contains, contains_repl, expr)

    pattern_index = r'(\w+)\.index\(((?:[^()]|\([^()]*\))*)\)'
    def index_repl(m):
        arr_name = m.group(1)
        value_str = m.group(2).strip()
        if arr_name not in variables or not isinstance(variables[arr_name], list):
            raise Exception(f"{arr_name}は配列ではありません")
        arr = variables[arr_name]
        try:
            v = clambon_eval(value_str, variables)
        except Exception:
            v = value_str
        try:
            idx = arr.index(v)
            return str(idx)
        except ValueError:
            return '0'
    prev_expr = None
    while prev_expr != expr:
        prev_expr = expr
        expr = re.sub(pattern_index, index_repl, expr)

    pattern_length = r'(\w+)\.length'
    def length_repl(m):
        var_name = m.group(1)
        if var_name not in variables:
            raise Exception(f"{var_name}が見つかりません")
        val = variables[var_name]
        if isinstance(val, (list, str)):
            return str(len(val))
        else:
            raise Exception(f"{var_name}は配列でも文字列でもありません")
    prev_expr = None
    while prev_expr != expr:
        prev_expr = expr
        expr = re.sub(pattern_length, length_repl, expr)

    pattern_letter = r'(\w+)\.letter\(([^)]+)\)'
    def letter_repl(m):
        var_name = m.group(1)
        idx_expr = m.group(2).strip()
        if var_name not in variables or not isinstance(variables[var_name], str):
            raise Exception(f"{var_name}は文字列ではありません")
        s = variables[var_name]
        try:
            idx = int(clambon_eval(idx_expr, variables))
            if 0 <= idx < len(s):
                return f'"{s[idx]}"'
            else:
                return '""'
        except Exception:
            return '""'
    prev_expr = None
    while prev_expr != expr:
        prev_expr = expr
        expr = re.sub(pattern_letter, letter_repl, expr)

    pattern_current = r'current\.(year|month|day|weekday|hour|minute|second)'
    def current_repl(m):
        t = time.localtime()
        weekday_sun0 = (t.tm_wday + 1) % 7
        mapping = {
            'year':   t.tm_year,
            'month':  t.tm_mon,
            'day':    t.tm_mday,
            'weekday': weekday_sun0,
            'hour':   t.tm_hour,
            'minute': t.tm_min,
            'second': t.tm_sec,
        }
        return str(mapping.get(m.group(1), 0))
    prev_expr = None
    while prev_expr != expr:
        prev_expr = expr
        expr = re.sub(pattern_current, current_repl, expr)

    def _handle_random(arg_s):
        parts = _split_args(arg_s)
        if len(parts) != 2:
            return "0"
        try:
            a = clambon_eval(parts[0], variables)
            b = clambon_eval(parts[1], variables)
            if not (_is_number(a) and _is_number(b)):
                return "0"
        except Exception:
            return "0"
        lo, hi = (a, b) if a <= b else (b, a)
        if float(a).is_integer() and float(b).is_integer():
            return str(random.randint(int(lo), int(hi)))
        else:
            return str(random.uniform(float(lo), float(hi)))

    def _handle_round(arg_s):
        try:
            x = clambon_eval(arg_s, variables)
            if not _is_number(x): return "0"
            x = float(x)
        except Exception:
            return "0"
        val = math.floor(x + 0.5) if x >= 0 else math.ceil(x - 0.5)
        return str(int(val))

    def _handle_abs(arg_s):
        try:
            v = clambon_eval(arg_s, variables)
            if not _is_number(v): return "0"
        except Exception:
            return "0"
        if isinstance(v, float) and not float(v).is_integer():
            return str(abs(v))
        else:
            return str(int(abs(v)))

    def _handle_floor(arg_s):
        try:
            v = clambon_eval(arg_s, variables)
            if not _is_number(v): return "0"
            return str(int(math.floor(float(v))))
        except Exception:
            return "0"

    def _handle_ceiling(arg_s):
        try:
            v = clambon_eval(arg_s, variables)
            if not _is_number(v): return "0"
            return str(int(math.ceil(float(v))))
        except Exception:
            return "0"

    def _handle_sqrt(arg_s):
        try:
            v = clambon_eval(arg_s, variables)
            if not _is_number(v): return "0"
            v = float(v)
            if v < 0: return "0"
            r = math.sqrt(v)
            return str(int(r)) if r.is_integer() else str(r)
        except Exception:
            return "0"

    def _handle_sin(arg_s):
        try:
            v = clambon_eval(arg_s, variables)
            if not _is_number(v): return "0"
            rad = math.radians(float(v))
            r = math.sin(rad)
            if abs(r) < 1e-15: r = 0.0
            return str(int(r)) if float(r).is_integer() else str(r)
        except Exception:
            return "0"

    def _handle_cos(arg_s):
        try:
            v = clambon_eval(arg_s, variables)
            if not _is_number(v): return "0"
            rad = math.radians(float(v))
            r = math.cos(rad)
            if abs(r) < 1e-15: r = 0.0
            return str(int(r)) if float(r).is_integer() else str(r)
        except Exception:
            return "0"

    def _handle_tan(arg_s):
        try:
            v = clambon_eval(arg_s, variables)
            if not _is_number(v): return "0"
            rad = math.radians(float(v))
            c = math.cos(rad)
            if abs(c) < 1e-15:
                return "0"
            r = math.tan(rad)
            if abs(r) < 1e-15: r = 0.0
            return str(int(r)) if float(r).is_integer() else str(r)
        except Exception:
            return "0"

    def _handle_asin(arg_s):
        try:
            v = clambon_eval(arg_s, variables)
            if not _is_number(v): return "0"
            v = float(v)
            if v < -1 or v > 1: return "0"
            r = math.degrees(math.asin(v))
            if abs(r) < 1e-15: r = 0.0
            return str(int(r)) if float(r).is_integer() else str(r)
        except Exception:
            return "0"

    def _handle_acos(arg_s):
        try:
            v = clambon_eval(arg_s, variables)
            if not _is_number(v): return "0"
            v = float(v)
            if v < -1 or v > 1: return "0"
            r = math.degrees(math.acos(v))
            if abs(r) < 1e-15: r = 0.0
            return str(int(r)) if float(r).is_integer() else str(r)
        except Exception:
            return "0"

    def _handle_atan(arg_s):
        try:
            v = clambon_eval(arg_s, variables)
            if not _is_number(v): return "0"
            r = math.degrees(math.atan(float(v)))
            if abs(r) < 1e-15: r = 0.0
            return str(int(r)) if float(r).is_integer() else str(r)
        except Exception:
            return "0"

    def _handle_ln(arg_s):
        try:
            v = clambon_eval(arg_s, variables)
            if not _is_number(v): return "0"
            v = float(v)
            if v <= 0: return "0"
            r = math.log(v)
            if abs(r) < 1e-15: r = 0.0
            return str(int(r)) if float(r).is_integer() else str(r)
        except Exception:
            return "0"

    def _handle_log(arg_s):
        try:
            v = clambon_eval(arg_s, variables)
            if not _is_number(v): return "0"
            v = float(v)
            if v <= 0: return "0"
            r = math.log10(v)
            if abs(r) < 1e-15: r = 0.0
            return str(int(r)) if float(r).is_integer() else str(r)
        except Exception:
            return "0"

    def _handle_exp(arg_s):
        try:
            v = clambon_eval(arg_s, variables)
            if not _is_number(v): return "0"
            r = math.exp(float(v))
            return str(int(r)) if float(r).is_integer() else str(r)
        except Exception:
            return "0"

    def _handle_pow10(arg_s):
        try:
            v = clambon_eval(arg_s, variables)
            if not _is_number(v): return "0"
            r = 10 ** float(v)
            return str(int(r)) if float(r).is_integer() else str(r)
        except Exception:
            return "0"

    def _handle_days_since_2000(arg_s):
        if arg_s.strip():
            print("daysSince2000は引数を取りません")
            return "0"
        secs = time.time() - 946684800.0
        days = secs / 86400.0
        return str(int(days)) if float(days).is_integer() else str(days)

    prev_expr = None
    while prev_expr != expr:
        prev_expr = expr
        expr = _replace_func(expr, 'random',  _handle_random)
        expr = _replace_func(expr, 'round',   _handle_round)
        expr = _replace_func(expr, 'abs',     _handle_abs)
        expr = _replace_func(expr, 'floor',   _handle_floor)
        expr = _replace_func(expr, 'ceiling', _handle_ceiling)
        expr = _replace_func(expr, 'sqrt',    _handle_sqrt)
        expr = _replace_func(expr, 'sin',     _handle_sin)
        expr = _replace_func(expr, 'cos',     _handle_cos)
        expr = _replace_func(expr, 'tan',     _handle_tan)
        expr = _replace_func(expr, 'asin',    _handle_asin)
        expr = _replace_func(expr, 'acos',    _handle_acos)
        expr = _replace_func(expr, 'atan',    _handle_atan)
        expr = _replace_func(expr, 'ln',      _handle_ln)
        expr = _replace_func(expr, 'log',     _handle_log)
        expr = _replace_func(expr, 'exp',     _handle_exp)
        expr = _replace_func(expr, 'pow10',   _handle_pow10)
        expr = _replace_func(expr, 'daysSince2000', _handle_days_since_2000)

    def _quote_double(s: str) -> str:
        return '"' + s.replace('\\', '\\\\').replace('"', '\\"') + '"'

    def _call_user_func(fname, arg_s):
        spec = variables['__funcs'][fname]
        args = _split_args(arg_s) if arg_s else []
        if len(args) != len(spec['params']):
            print(f"引数の個数が一致しません: 期待 {len(spec['params'])}, 実際 {len(args)}")
            return "0"
        vals = [clambon_eval(a, variables) for a in args]
        backups = {p: variables.get(p, None) for p in spec['params']}
        exists  = {p: (p in variables) for p in spec['params']}
        for p, v in zip(spec['params'], vals): variables[p] = v
        variables['__call_depth'] += 1
        try:
            run_clambon(spec['body'], variables)
            print(f"このブロックは戻り値がありません（式では使えません）: {fname}")
            return "0"
        except ReturnSignal as rs:
            ret = rs.value
            if isinstance(ret, bool):
                token = "True" if ret else "False"
            elif isinstance(ret, (int, float)):
                token = str(int(ret)) if (isinstance(ret, float) and float(ret).is_integer()) else str(ret)
            elif isinstance(ret, str):
                token = _quote_double(ret)
            else:
                print("returnはnumber/bool/stringのみです"); token = "0"
            return token
        finally:
            variables['__call_depth'] -= 1
            for p in spec['params']:
                if exists[p]: variables[p] = backups[p]
                else: variables.pop(p, None)

    if '__funcs' in variables and variables['__funcs']:
        for _fname in list(variables['__funcs'].keys()):
            expr = _replace_func(expr, _fname, lambda a, _f=_fname: _call_user_func(_f, a))

    expr = _replace_word_outside_strings(expr, 'mod', '%')

    _ensure_only_supported_comparators(expr)
    return eval(expr, {}, variables)

def run_clambon(source_code, variables=None):
    if variables is None:
        variables = {}

    if '__timer_start' not in variables:
        variables['__timer_start'] = time.time()
        variables['timer'] = 0

    variables.setdefault('__funcs', {})
    variables.setdefault('__call_depth', 0)

    source_code = source_code.replace('} else {', '}\nelse {')

    lines = source_code.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if '//' in line:
            line = line.split('//', 1)[0].rstrip()

        if line == '' or line.startswith('//'):
            i += 1
            continue

        if line.startswith('import '):
            module_name = line[len('import '):].strip()
            load_extension(module_name)
            i += 1
            continue

        line = process_extensions(line)

        if line.startswith('def '):
            if '{' not in line:
                print(f"def構文エラー: {line}"); i += 1; continue
            header = line[4:line.find('{')].strip()
            if '(' not in header or ')' not in header:
                print(f"def構文エラー: {line}"); i += 1; continue
            fname = header[:header.find('(')].strip()
            params = [p.strip() for p in header[header.find('(')+1:header.rfind(')')].split(',') if p.strip()]
            block = []
            i += 1; brace = 1
            while i < len(lines) and brace > 0:
                l = lines[i]
                if '{' in l: brace += l.count('{')
                if '}' in l: brace -= l.count('}')
                if brace > 0: block.append(l)
                i += 1
            if fname in variables['__funcs']:
                print(f"{fname} を再定義しました")
            variables['__funcs'][fname] = {'params': params, 'body': '\n'.join(block)}
            continue

        if line.startswith('return'):
            if variables.get('__call_depth', 0) <= 0:
                print("returnはブロック内でのみ使用できます")
                i += 1
                continue
            expr = line[6:].strip()
            if not expr:
                print("returnの式が必要です")
                raise ReturnSignal(0)
            val = clambon_eval(expr, variables)
            if not isinstance(val, (int, float, bool, str)):
                print("returnはnumber/bool/stringのみです"); val = 0
            raise ReturnSignal(val)

        m = re.match(r'^(\w+)\[(.+)\]\s*=\s*(.+)$', line)
        if m:
            arr_name = m.group(1)
            idx_str = m.group(2).strip()
            value_str = m.group(3).strip()
            if arr_name not in variables or not isinstance(variables[arr_name], list):
                print(f"{arr_name}は配列ではありません")
            else:
                arr = variables[arr_name]
                try:
                    idx = clambon_eval(idx_str, variables)
                    if not isinstance(idx, int):
                        raise ValueError
                except Exception:
                    print(f"インデックスが不正です: {idx_str}")
                    i += 1
                    continue
                try:
                    v = clambon_eval(value_str, variables)
                except Exception:
                    v = value_str
                if 0 <= idx < len(arr):
                    arr[idx] = v
                else:
                    print(f"インデックス範囲外です: {idx}")
            i += 1
            continue

        if '.add(' in line and line.endswith(')'):
            try:
                arr_name, value = line.split('.add(', 1)
                arr_name = arr_name.strip()
                value = value[:-1].strip()
                if arr_name not in variables or not isinstance(variables[arr_name], list):
                    print(f"addエラー: {arr_name}は配列ではありません")
                else:
                    arr = variables[arr_name]
                    try:
                        v = clambon_eval(value.replace('true', 'True').replace('false', 'False'), variables)
                    except Exception:
                        v = value
                    if len(arr) == 0 or _same_clam_type(arr[0], v):
                        arr.append(v)
                    else:
                        print(f"add型エラー: {arr_name}にはその型は追加できません")
            except Exception:
                print(f"add構文エラー: {line}")
            i += 1
            continue

        if '.insert(' in line and ', at:' in line and line.endswith(')'):
            try:
                prefix, rest = line.split('.insert(', 1)
                value_part, index_part = rest.split(', at:')
                value_str = value_part.strip()
                index_str = index_part[:-1].strip()
                arr_name = prefix.strip()
                if arr_name not in variables or not isinstance(variables[arr_name], list):
                    print(f"insertエラー: {arr_name}は配列ではありません")
                else:
                    arr = variables[arr_name]
                    try:
                        v = clambon_eval(value_str.replace('true', 'True').replace('false', 'False'), variables)
                    except Exception:
                        v = value_str
                    try:
                        idx = clambon_eval(index_str, variables)
                        if not isinstance(idx, int):
                            raise ValueError
                    except Exception:
                        print(f"insertエラー: インデックスが不正です: {index_str}")
                        i += 1
                        continue
                    if len(arr) == 0 or _same_clam_type(arr[0], v):
                        if 0 <= idx <= len(arr):
                            arr.insert(idx, v)
                        else:
                            print(f"insertエラー: インデックス範囲外です")
                    else:
                        print(f"insert型エラー: {arr_name}にはその型は挿入できません")
            except Exception:
                print(f"insert構文エラー: {line}")
            i += 1
            continue

        if '.delete(' in line and line.endswith(')'):
            try:
                arr_name, value = line.split('.delete(', 1)
                arr_name = arr_name.strip()
                value = value[:-1].strip()
                if arr_name not in variables or not isinstance(variables[arr_name], list):
                    print(f"deleteエラー: {arr_name}は配列ではありません")
                else:
                    arr = variables[arr_name]
                    if value == 'all':
                        arr.clear()
                    else:
                        try:
                            idx = int(value)
                            if 0 <= idx < len(arr):
                                del arr[idx]
                            else:
                                print(f"deleteエラー: インデックス範囲外です")
                        except Exception:
                            print(f"deleteエラー: インデックスが不正です")
            except Exception:
                print(f"delete構文エラー: {line}")
            i += 1
            continue

        if line.startswith('wait(') and line.endswith(')'):
            arg = line[len('wait('):-1].strip()
            try:
                val = clambon_eval(arg, variables)
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    time.sleep(val)
                else:
                    while True:
                        cond = clambon_eval(arg, variables)
                        if cond:
                            break
                        time.sleep(0.05)
            except Exception as e:
                print(f"waitの引数エラー: {arg} ({e})")
            i += 1
            continue

        if line.startswith('resetTimer(') and line.endswith(')'):
            arg = line[len('resetTimer('):-1].strip()
            if arg != '':
                print("resetTimerは引数を取りません")
            else:
                variables['__timer_start'] = time.time()
                variables['timer'] = 0
            i += 1
            continue

        m = re.match(r'^([A-Za-z_]\w*)\((.*)\)$', line)
        if m and m.group(1) in variables['__funcs']:
            fname = m.group(1)
            arg_s = m.group(2).strip()
            spec = variables['__funcs'][fname]
            args = _split_args(arg_s) if arg_s else []
            if len(args) != len(spec['params']):
                print(f"引数の個数が一致しません: 期待 {len(spec['params'])}, 実際 {len(args)}")
                i += 1; continue
            vals = [clambon_eval(a, variables) for a in args]
            backups = {p: variables.get(p, None) for p in spec['params']}
            exists  = {p: (p in variables) for p in spec['params']}
            for p, v in zip(spec['params'], vals): variables[p] = v
            variables['__call_depth'] += 1
            try:
                run_clambon(spec['body'], variables)
            except ReturnSignal:
                pass
            finally:
                variables['__call_depth'] -= 1
                for p in spec['params']:
                    if exists[p]: variables[p] = backups[p]
                    else: variables.pop(p, None)
            i += 1
            continue

        if line.startswith('var '):
            try:
                _, rest = line.split('var ', 1)
                var, value = rest.split('=', 1)
                var = var.strip()
                value = value.strip()

                if var == 'timer':
                    print("timerは予約語のため宣言できません")
                    i += 1
                    continue

                if (re.match(r'^\([^\)]*\)$', value) or (value.startswith('(') and value.endswith(')'))
                   or re.match(r'^\w+\.index\(', value)
                   or re.match(r'^\w+\.length$', value)
                   or re.match(r'^\w+\.contains\(', value)
                   or re.match(r'^\w+\.letter\(', value)
                   or value.startswith('random(')
                   or value.startswith('round(')
                   or value.startswith('abs(')
                   or value.startswith('floor(')
                   or value.startswith('ceiling(')
                   or value.startswith('sqrt(')
                   or value.startswith('sin(')
                   or value.startswith('cos(')
                   or value.startswith('tan(')
                   or value.startswith('asin(')
                   or value.startswith('acos(')
                   or value.startswith('atan(')
                   or value.startswith('ln(')
                   or value.startswith('log(')
                   or value.startswith('exp(')
                   or value.startswith('pow10(')
                   or _FUNC_CALL_RE.match(value)
                   or value.startswith('current.')
                   or value == 'timer'):
                    try:
                        result = clambon_eval(value, variables)
                        variables[var] = result
                    except Exception:
                        print(f"計算エラー: {value}")
                elif value.startswith('[') and value.endswith(']'):
                    value_py = value.replace('true', 'True').replace('false', 'False')
                    try:
                        result = clambon_eval(value_py, variables)
                        if len(result) > 0:
                            head = result[0]
                            for v in result[1:]:
                                if not _same_clam_type(head, v):
                                    print(f"配列型エラー: {value}")
                                    result = []
                                    break
                        variables[var] = result
                    except Exception:
                        print(f"配列エラー: {value}")
                elif value.startswith('"') and value.endswith('"'):
                    variables[var] = value[1:-1]
                elif value == 'true':
                    variables[var] = True
                elif value == 'false':
                    variables[var] = False
                elif _NUMERIC_RE.match(value):
                    variables[var] = float(value) if '.' in value else int(value)
                else:
                    print(f"サポートされていない（カッコで囲まれていない）値です: {value}")
            except Exception:
                print(f"構文エラー: {line}")
            i += 1
            continue

        if line.startswith('repeat '):
            cond_start = line.find('repeat ') + 7
            cond_end = line.find('{')
            cond_str = line[cond_start:cond_end].strip()
            if not (cond_str.startswith('(') and cond_str.endswith(')')):
                print("repeat文の条件式は必ず()で囲んでください")
                i += 1
                continue
            block = []
            i += 1
            brace_count = 1
            while i < len(lines) and brace_count > 0:
                l = lines[i]
                if '{' in l: brace_count += l.count('{')
                if '}' in l: brace_count -= l.count('}')
                if brace_count > 0: block.append(l)
                i += 1
            try:
                while True:
                    cond_eval = clambon_bool_replacer(cond_str, variables)
                    if clambon_eval(cond_eval, variables):
                        break
                    run_clambon('\n'.join(block), variables)
            except ReturnSignal:
                raise
            except Exception:
                print(f"repeat条件式エラー: {cond_str}")
            continue

        if line.startswith('if '):
            cond_start = line.find('if ') + 3
            cond_end = line.find('{')
            cond_str = line[cond_start:cond_end].strip()
            if not (cond_str.startswith('()') or (cond_str.startswith('(') and cond_str.endswith(')'))):
                print("if文の条件式は必ず()で囲んでください")
                brace_count = 1
                i += 1
                while i < len(lines) and brace_count > 0:
                    if '{' in lines[i]: brace_count += lines[i].count('{')
                    if '}' in lines[i]: brace_count -= lines[i].count('}')
                    i += 1
                continue
            cond_str = clambon_bool_replacer(cond_str, variables)
            block = []
            i += 1
            brace_count = 1
            while i < len(lines) and brace_count > 0:
                l = lines[i]
                if '{' in l: brace_count += l.count('{')
                if '}' in l: brace_count -= l.count('}')
                if brace_count > 0: block.append(l)
                i += 1
            else_block = []
            if i < len(lines) and lines[i].strip().startswith('else'):
                i += 1
                brace_count = 1
                while i < len(lines) and brace_count > 0:
                    l = lines[i]
                    if '{' in l: brace_count += l.count('{')
                    if '}' in l: brace_count -= l.count('}')
                    if brace_count > 0: else_block.append(l)
                    i += 1
            try:
                if clambon_eval(cond_str, variables):
                    run_clambon('\n'.join(block), variables)
                elif else_block:
                    run_clambon('\n'.join(else_block), variables)
            except ReturnSignal:
                raise
            except Exception:
                print(f"if条件式エラー: {cond_str}")
            continue

        if line.startswith('while '):
            cond_start = line.find('while ') + 6
            cond_end = line.find('{')
            cond_str = line[cond_start:cond_end].strip()
            if not (cond_str.startswith('(') and cond_str.endswith(')')):
                print("while文の条件式は必ず()で囲んでください")
                brace_count = 1
                i += 1
                while i < len(lines) and brace_count > 0:
                    if '{' in lines[i]: brace_count += lines[i].count('{')
                    if '}' in lines[i]: brace_count -= lines[i].count('}')
                    i += 1
                continue
            cond_str = clambon_bool_replacer(cond_str, variables)
            block = []
            i += 1
            brace_count = 1
            while i < len(lines) and brace_count > 0:
                l = lines[i]
                if '{' in l: brace_count += l.count('{')
                if '}' in l: brace_count -= l.count('}')
                if brace_count > 0: block.append(l)
                i += 1
            try:
                while True:
                    cond_eval = clambon_bool_replacer(cond_str, variables)
                    if not clambon_eval(cond_eval, variables):
                        break
                    run_clambon('\n'.join(block), variables)
            except ReturnSignal:
                raise
            except Exception:
                print(f"while条件式エラー: {cond_str}")
            continue

        if line.startswith('for '):
            if '{' not in line:
                print(f"for文の構文エラー: {line}")
                i += 1
                continue
            for_header = line[4:line.find('{')].strip()
            if ' in ' not in for_header:
                print(f"for文の構文エラー: {line}")
                i += 1
                continue
            var, count = [s.strip() for s in for_header.split(' in ', 1)]
            if var == '':
                var = '_'
            if not (count.startswith('(') and count.endswith(')')):
                print("for文の回数式は必ず()で囲んでください")
                brace_count = 1
                i += 1
                while i < len(lines) and brace_count > 0:
                    if '{' in lines[i]: brace_count += lines[i].count('{')
                    if '}' in lines[i]: brace_count -= lines[i].count('}')
                    i += 1
                continue
            block = []
            i += 1
            brace_count = 1
            while i < len(lines) and brace_count > 0:
                l = lines[i]
                if '{' in l: brace_count += l.count('{')
                if '}' in l: brace_count -= l.count('}')
                if brace_count > 0: block.append(l)
                i += 1
            try:
                n = clambon_eval(count, variables)
                for idx in range(int(n)):
                    variables[var] = idx
                    run_clambon('\n'.join(block), variables)
            except ReturnSignal:
                raise
            except Exception:
                print(f"for文エラー: {line}")
            continue

        if line.startswith('forever {') or line == 'forever{':
            block = []
            i += 1
            brace_count = 1
            while i < len(lines) and brace_count > 0:
                l = lines[i]
                if '{' in l: brace_count += l.count('{')
                if '}' in l: brace_count -= l.count('}')
                if brace_count > 0: block.append(l)
                i += 1
            try:
                while True:
                    run_clambon('\n'.join(block), variables)
            except ReturnSignal:
                raise
            except Exception:
                print(f"forever文エラー: {line}")
            continue

        if '=' in line and not line.startswith('var '):
            try:
                var, value = line.split('=', 1)
                var = var.strip()
                value = value.strip()

                if var == 'timer':
                    print("timerは予約語のため代入できません")
                    i += 1
                    continue

                if (re.match(r'^\([^\)]*\)$', value) or (value.startswith('(') and value.endswith(')'))
                   or re.match(r'^\w+\.index\(', value)
                   or re.match(r'^\w+\.length$', value)
                   or re.match(r'^\w+\.contains\(', value)
                   or re.match(r'^\w+\.letter\(', value)
                   or value.startswith('random(')
                   or value.startswith('round(')
                   or value.startswith('abs(')
                   or value.startswith('floor(')
                   or value.startswith('ceiling(')
                   or value.startswith('sqrt(')
                   or value.startswith('sin(')
                   or value.startswith('cos(')
                   or value.startswith('tan(')
                   or value.startswith('asin(')
                   or value.startswith('acos(')
                   or value.startswith('atan(')
                   or value.startswith('ln(')
                   or value.startswith('log(')
                   or value.startswith('exp(')
                   or value.startswith('pow10(')
                   or _FUNC_CALL_RE.match(value)
                   or value.startswith('current.')
                   or value == 'timer'):
                    result = clambon_eval(value, variables)
                    variables[var] = result
                elif value.startswith('"') and value.endswith('"'):
                    variables[var] = value[1:-1]
                elif value == 'true':
                    variables[var] = True
                elif value == 'false':
                    variables[var] = False
                elif _NUMERIC_RE.match(value):
                    variables[var] = float(value) if '.' in value else int(value)
                else:
                    print(f"再代入: カッコで囲まれていない値はサポートされていません: {value}")
            except Exception:
                print(f"代入エラー: {line}")
            i += 1
            continue

        if line.startswith('print(') and line.endswith(')'):
            arg = line[len('print('):-1].strip()
            if arg == 'current':
                print("currentは単体では使用できません（例: current.year）")
                i += 1
                continue
            try:
                result = clambon_eval(arg, variables)
                if isinstance(result, bool):
                    print('true' if result else 'false')
                else:
                    print(result)
            except Exception:
                print(f"printの式エラー: {arg}")
            i += 1
            continue

        m = re.match(r'^([A-Za-z_]\w*)\s*\(.*\)$', line)
        if m and m.group(1) not in variables.get('__funcs', {}):
            print(f"未定義のブロックです: {m.group(1)}")
            i += 1
            continue

        if line == "" or line == "}":
            i += 1
            continue
        else:
            print(f"サポートされていない構文です: {line}")
            i += 1

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Clambonコードのファイル名を指定してください")
    else:
        filename = sys.argv[1]
        if not filename.endswith('.clam'):
            print(".clam拡張子のファイルのみ実行できます")
            raise SystemExit(1)
        with open(filename, 'r', encoding='utf-8') as f:
            code = f.read()
        run_clambon(code)
