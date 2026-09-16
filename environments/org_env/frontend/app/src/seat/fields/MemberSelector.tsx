// Member selector with role display and online status
import { useState, useRef, useEffect } from "react";

type Member = {
  agent_id: string;
  name: string;
  role: string;
  online?: boolean;
  identity?: string;
};

type Props = {
  value: string;
  onChange: (value: string) => void;
  members: Member[];
  placeholder?: string;
  multiple?: boolean;
};

export function MemberSelector({ value, onChange, members, placeholder, multiple }: Props) {
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState("");
  const [focused, setFocused] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  // Handle multiple selection (comma-separated)
  const values = multiple ? value.split(",").map((v) => v.trim()).filter(Boolean) : [value];
  const current = members.filter((m) => values.includes(m.agent_id) || values.includes(m.name));

  // Fuzzy filter members by search term
  const filtered = members.filter((m) => {
    if (!search) return true;
    const q = search.toLowerCase();
    return (
      m.name.toLowerCase().includes(q) ||
      m.role.toLowerCase().includes(q) ||
      m.agent_id.toLowerCase().includes(q)
    );
  }).slice(0, 10);

  useEffect(() => {
    if (open && filtered.length > 0) {
      setFocused(0);
    }
  }, [open, filtered.length]);

  const select = (member: Member) => {
    if (multiple) {
      const newValues = [...values, member.name];
      onChange(newValues.join(", "));
      setSearch("");
      inputRef.current?.focus();
    } else {
      onChange(member.agent_id);
      setSearch("");
      setOpen(false);
      inputRef.current?.blur();
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setFocused((f) => Math.min(f + 1, filtered.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setFocused((f) => Math.max(f - 1, 0));
    } else if (e.key === "Enter" && filtered.length > 0) {
      e.preventDefault();
      select(filtered[focused]);
    } else if (e.key === "Escape") {
      setOpen(false);
      setSearch("");
    }
  };

  const displayValue = () => {
    if (open) return search;
    if (multiple && current.length > 0) {
      return current.map((m) => m.name).join(", ");
    }
    if (current.length === 1) {
      return `${current[0].name} (${current[0].role.replace(/_/g, " ")})`;
    }
    return value;
  };

  return (
    <div className="smart-field member-selector">
      <input
        ref={inputRef}
        type="text"
        value={displayValue()}
        onChange={(e) => {
          setSearch(e.target.value);
          if (!open) setOpen(true);
        }}
        onFocus={() => setOpen(true)}
        onBlur={() => setTimeout(() => setOpen(false), 200)}
        onKeyDown={handleKeyDown}
        placeholder={placeholder || (multiple ? "Type @ to mention..." : "Search members...")}
      />
      {open && filtered.length > 0 && (
        <div className="smart-field-dropdown">
          {filtered.map((member, i) => (
            <button
              key={member.agent_id}
              className={`dropdown-item ${i === focused ? "focused" : ""}`}
              onClick={() => select(member)}
              onMouseEnter={() => setFocused(i)}
            >
              <div className="member-item">
                <span className={`member-dot ${member.online ? "online" : "offline"}`} />
                <span className="member-name">{member.name}</span>
                <span className="member-role">{member.role.replace(/_/g, " ")}</span>
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
