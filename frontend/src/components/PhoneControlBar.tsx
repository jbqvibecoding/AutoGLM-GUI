import React, { useState } from 'react';
import { ArrowLeft, CornerDownLeft, Delete, House, Send } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { sendKey, sendText, type PhoneKey } from '../api';
import { useTranslation } from '../lib/i18n-context';

interface PhoneControlBarProps {
  deviceId: string;
  /** Remote device agents only support Back, Home and text. */
  isRemoteDevice: boolean;
  visible: boolean;
  onResult: (message: string, isError: boolean) => void;
}

/** Keys and text input for taking over the phone by hand. */
export function PhoneControlBar({
  deviceId,
  isRemoteDevice,
  visible,
  onResult,
}: PhoneControlBarProps) {
  const t = useTranslation();
  const [text, setText] = useState('');
  const [sending, setSending] = useState(false);
  const [focused, setFocused] = useState(false);

  const fail = (error?: string | null) =>
    onResult(
      (t.devicePanel.tapError || 'Error: {error}').replace(
        '{error}',
        error || 'unknown'
      ),
      true
    );

  const press = async (key: PhoneKey) => {
    try {
      const res = await sendKey(deviceId, key);
      if (!res.success) fail(res.error);
    } catch (error) {
      fail(String(error));
    }
  };

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!text || sending) return;
    setSending(true);
    try {
      const res = await sendText(deviceId, text);
      if (res.success) {
        setText('');
        onResult(t.devicePanel.textSent, false);
      } else {
        fail(res.error);
      }
    } catch (error) {
      fail(String(error));
    } finally {
      setSending(false);
    }
  };

  const keys: { key: PhoneKey; label: string; icon: React.ReactNode }[] = [
    { key: 'back', label: t.devicePanel.keyBack, icon: <ArrowLeft /> },
    { key: 'home', label: t.devicePanel.keyHome, icon: <House /> },
    ...(isRemoteDevice
      ? []
      : [
          {
            key: 'enter' as const,
            label: t.devicePanel.keyEnter,
            icon: <CornerDownLeft />,
          },
          {
            key: 'delete' as const,
            label: t.devicePanel.keyDelete,
            icon: <Delete />,
          },
        ]),
  ];

  const shown = visible || focused || text.length > 0;

  return (
    <div
      className={`absolute bottom-3 left-3 right-3 z-20 transition-opacity duration-200 ${
        shown ? 'opacity-100' : 'opacity-0 pointer-events-none'
      }`}
    >
      <div className="flex items-center gap-1 rounded-xl border bg-background/90 p-1.5 shadow-lg backdrop-blur-sm">
        {keys.map(({ key, label, icon }) => (
          <Button
            key={key}
            type="button"
            size="icon"
            variant="ghost"
            className="h-8 w-8 flex-shrink-0"
            title={label}
            aria-label={label}
            onClick={() => press(key)}
          >
            {icon}
          </Button>
        ))}
        <form
          onSubmit={submit}
          className="flex min-w-0 flex-1 items-center gap-1"
        >
          <Input
            value={text}
            onChange={event => setText(event.target.value)}
            onFocus={() => setFocused(true)}
            onBlur={() => setFocused(false)}
            placeholder={t.devicePanel.typeOnPhone}
            aria-label={t.devicePanel.typeOnPhone}
            autoComplete="off"
            maxLength={2000}
            className="h-8 min-w-0"
          />
          <Button
            type="submit"
            size="icon"
            className="h-8 w-8 flex-shrink-0"
            disabled={!text || sending}
            title={t.devicePanel.sendText}
            aria-label={t.devicePanel.sendText}
          >
            <Send />
          </Button>
        </form>
      </div>
    </div>
  );
}
