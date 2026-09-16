// Rich text input with @mention autocomplete support
import { useState, useRef, useEffect } from "react";

type Member = {
  agent_id: string;
  name: string;
  role: string;
};

type Props = {
  value: string;
  onChange: (value: string) => void;
  members: Member[];
  placeholder?: string;
  rows?: number;
};

export function RichTextInput({ value, onChange, members, placeholder, rows = 3 }: Props) {
  const [mentionSearch, setMentionSearch] = useState<string | null>(null);
  const [mentionPos, setMentionPos] = useState(0);
  const [focused, setFocused] = useState(0);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Extract @mentions from text
  const extractMentions = (text: string): string[] => {
    const mentions = text.match(/@(\w+)/g);
    return mentions ? mentions.map((m) => m.slice(1)) : [];
  };

  // Filter members based on search after @
  const filtered = mentionSearch
    ? members.filter((m) =>
        m.name.toLowerCase().startsWith(mentionSearch.toLowerCase())
      ).slice(0, 5)
    : [];

  useEffect(() => {
    if (filtered.length > 0) {
      setFocused(0);
    }
  }, [filtered.length]);

  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const newValue = e.target.value;
    const cursorPos = e.target.selectionStart;

    onChange(newValue);

    // Check if we're typing after @
    const beforeCursor = newValue.slice(0, cursorPos);
    const lastAt = beforeCursor.lastIndexOf("@");

    if (lastAt !== -1) {
      const afterAt = beforeCursor.slice(lastAt + 1);
      // Only show mentions if @ is followed by word chars (no space)
      if (/^\w*$/.test(afterAt)) {
        setMentionSearch(afterAt);
        setMentionPos(lastAt);
        return;
      }
    }

    setMentionSearch(null);
  };

  const insertMention = (member: Member) => {
    if (mentionPos === null) return;

    const before = value.slice(0, mentionPos);
    const after = value.slice(textareaRef.current?.selectionStart || value.length);
    const newValue = `${before}@${member.name} ${after}`;

    onChange(newValue);
    setMentionSearch(null);

    // Focus back and move cursor
    setTimeout(() => {
      textareaRef.current?.focus();
      const newPos = mentionPos + member.name.length + 2;
      textareaRef.current?.setSelectionRange(newPos, newPos);
    }, 0);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (!mentionSearch || filtered.length === 0) return;

    if (e.key === "ArrowDown") {
      e.preventDefault();
      setFocused((f) => Math.min(f + 1, filtered.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setFocused((f) => Math.max(f - 1, 0));
    } else if (e.key === "Enter" || e.key === "Tab") {
      e.preventDefault();
      insertMention(filtered[focused]);
    } else if (e.key === "Escape") {
      setMentionSearch(null);
    }
  };

  return (
    <div className="smart-field rich-text-input">
      <textarea
        ref={textareaRef}
        rows={rows}
        value={value}
        onChange={handleChange}
        onKeyDown={handleKeyDown}
        placeholder={placeholder || "Type @ to mention someone..."}
      />
      {mentionSearch !== null && filtered.length > 0 && (
        <div className="smart-field-dropdown mention-dropdown">
          {filtered.map((member, i) => (
            <button
              key={member.agent_id}
              className={`dropdown-item ${i === focused ? "focused" : ""}`}
              onClick={() => insertMention(member)}
              onMouseEnter={() => setFocused(i)}
            >
              <div className="member-item">
                <span className="member-name">@{member.name}</span>
                <span className="member-role">{member.role.replace(/_/g, " ")}</span>
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
